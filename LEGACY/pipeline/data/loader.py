"""Data loading from FMP SQLite database.

Loads raw financial data, validates completeness and schema consistency,
and returns a RawFinancialData contract object.

All column definitions are read from AnalysisConfig — no financial table
or column names are hardcoded in this module.

Adapted from algorithmic-investing/data/loader.py.
"""

import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pipeline.config.settings import AnalysisConfig, TableSpec
from pipeline.data.contracts import RawFinancialData

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, context: str) -> None:
    """Validate that a SQL identifier is safe (alphanumeric + underscore only).

    Raises:
        ValueError: If name contains characters outside [a-zA-Z0-9_].
    """
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(
            f"Unsafe SQL identifier in {context}: {name!r}. "
            f"Must match [a-zA-Z_][a-zA-Z0-9_]*."
        )


# Temporal columns present on every financial statement table.
# These are structural (database schema), not configurable financial metrics.
_TEMPORAL_DB_COLS: tuple[str, ...] = ("entityId", "fiscalYear", "period", "date")
_TEMPORAL_RENAMES: dict[str, str] = {
    "fiscalYear": "fiscal_year",
    "period": "period",
    "date": "date",
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection to the FMP database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A read-only sqlite3.Connection.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"FMP database not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_columns(
    conn: sqlite3.Connection,
    table: str,
    required_cols: list[str],
) -> None:
    """Verify that required columns exist in a database table.

    Raises:
        RuntimeError: If any required column is missing.
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    existing = {row["name"] for row in cursor.fetchall()}
    missing = set(required_cols) - existing
    if missing:
        raise RuntimeError(
            f"Missing required columns in '{table}': {sorted(missing)}. "
            f"The collector may not have populated this data."
        )


# ---------------------------------------------------------------------------
# Column requirements from config
# ---------------------------------------------------------------------------


def _get_column_requirements(
    config: AnalysisConfig,
) -> tuple[set[str], set[str]]:
    """Derive required and optional column sets from config.

    Returns:
        Tuple of (required_internal_names, optional_internal_names).
    """
    required: set[str] = set()
    optional: set[str] = set()

    for spec in config.table_specs:
        for col in spec.columns:
            if col.required:
                required.add(col.internal_name)
            else:
                optional.add(col.internal_name)

    if config.price_column.required:
        required.add(config.price_column.internal_name)
    else:
        optional.add(config.price_column.internal_name)

    return required, optional


# ---------------------------------------------------------------------------
# Entity loading
# ---------------------------------------------------------------------------


def _load_eligible_entities(
    conn: sqlite3.Connection,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Load entities passing all filters.

    Filters applied at query time:
    - isEtf = 0, isAdr = 0, isFund = 0, isActivelyTrading = 1
    - exchange in configured list

    Args:
        conn: Database connection.
        config: Pipeline configuration.

    Returns:
        DataFrame with entity metadata columns.
    """
    db_cols = [col.db_column for col in config.entity_columns]
    _validate_columns(
        conn, "entity",
        [*db_cols, "isEtf", "isAdr", "isFund", "isActivelyTrading"],
    )

    placeholders = ", ".join("?" for _ in config.exchanges)
    col_list = ", ".join(db_cols)
    query = f"""
        SELECT {col_list}
        FROM entity
        WHERE isEtf = 0
          AND isAdr = 0
          AND isFund = 0
          AND isActivelyTrading = 1
          AND exchange IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=list(config.exchanges))
    rename_map = {col.db_column: col.internal_name for col in config.entity_columns}
    df = df.rename(columns=rename_map)
    logger.info("Loaded %d eligible entities", len(df))
    return df


# ---------------------------------------------------------------------------
# Financial data loading
# ---------------------------------------------------------------------------


def _load_table_data(
    conn: sqlite3.Connection,
    spec: TableSpec,
    entity_ids: list[int],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Load required columns from a financial table per its TableSpec.

    Args:
        conn: Database connection.
        spec: Table specification from config.
        entity_ids: Eligible entity IDs.
        config: Pipeline configuration.

    Returns:
        DataFrame with temporal columns and renamed financial columns.

    Raises:
        RuntimeError: If required columns are missing from the table.
    """
    db_cols = [col.db_column for col in spec.columns]
    _validate_columns(conn, spec.table_name, list(_TEMPORAL_DB_COLS) + db_cols)

    all_cols = list(_TEMPORAL_DB_COLS) + db_cols
    col_list = ", ".join(all_cols)
    placeholders = ", ".join("?" for _ in entity_ids)

    # "FQ" is a pipeline-level shorthand for all quarters; the DB stores
    # individual quarter codes (Q1-Q4). "FY" maps directly.
    if config.period_type == "FQ":
        period_filter = "AND period IN ('Q1', 'Q2', 'Q3', 'Q4')"
        period_params: list[Any] = []
    else:
        period_filter = "AND period = ?"
        period_params = [config.period_type]

    query = f"""
        SELECT {col_list}
        FROM {spec.table_name}
        WHERE entityId IN ({placeholders})
          AND fiscalYear BETWEEN ? AND ?
          {period_filter}
    """
    params: list[Any] = [
        *entity_ids,
        config.min_fiscal_year,
        config.max_fiscal_year,
        *period_params,
    ]
    df = pd.read_sql_query(query, conn, params=params)

    rename_map: dict[str, str] = {
        **_TEMPORAL_RENAMES,
        "entityId": "entity_id",
        **{col.db_column: col.internal_name for col in spec.columns},
    }
    df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Price alignment
# ---------------------------------------------------------------------------


def _align_prices(
    conn: sqlite3.Connection,
    statement_dates: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Align eodPrice to fiscal period end dates.

    For each (entity_id, date) pair from financial statements, finds the
    nearest trading day price within a +/-7-day window. If no price exists
    within the window, the value is NaN.

    Args:
        conn: Database connection.
        statement_dates: DataFrame with columns [entity_id, date].
        config: Pipeline configuration.

    Returns:
        DataFrame with columns [entity_id, date, {price_internal_name}].
    """
    db_col = config.price_column.db_column
    internal_name = config.price_column.internal_name
    _validate_columns(conn, "eodPrice", ["entityId", "date", db_col])

    unique_dates = (
        statement_dates[["entity_id", "date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if unique_dates.empty:
        return pd.DataFrame(columns=["entity_id", "date", internal_name])

    # Temp table for efficient date-range joining.
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS _stmt_dates "
        "(entityId INTEGER, stmt_date TEXT)"
    )
    conn.execute("DELETE FROM _stmt_dates")
    conn.executemany(
        "INSERT INTO _stmt_dates (entityId, stmt_date) VALUES (?, ?)",
        list(zip(
            unique_dates["entity_id"].tolist(),
            unique_dates["date"].tolist(),
            strict=True,
        )),
    )

    # Find nearest trading day within +/-N days using window function.
    days = config.price_alignment_days
    query = f"""
        SELECT entity_id, date, {internal_name}
        FROM (
            SELECT
                sd.entityId AS entity_id,
                sd.stmt_date AS date,
                ep.{db_col} AS {internal_name},
                ROW_NUMBER() OVER (
                    PARTITION BY sd.entityId, sd.stmt_date
                    ORDER BY ABS(julianday(ep.date) - julianday(sd.stmt_date))
                ) AS rn
            FROM _stmt_dates sd
            LEFT JOIN eodPrice ep
                ON ep.entityId = sd.entityId
                AND ep.date BETWEEN date(sd.stmt_date, '-{days} days')
                              AND date(sd.stmt_date, '+{days} days')
            WHERE ep.{db_col} IS NOT NULL
        ) t WHERE rn = 1
    """

    prices = pd.read_sql_query(query, conn)
    conn.execute("DROP TABLE IF EXISTS _stmt_dates")

    # Re-merge to ensure all statement dates are present (NaN where no price).
    result = unique_dates.merge(prices, on=["entity_id", "date"], how="left")
    no_price_count = result[internal_name].isna().sum()
    if no_price_count > 0:
        logger.warning(
            "%d statement dates have no price within +/-7 days", no_price_count,
        )
    return result


# ---------------------------------------------------------------------------
# Joining with diagnostics
# ---------------------------------------------------------------------------


def _build_entity_symbol_map(entities: pd.DataFrame) -> dict[int, str]:
    """Build entity_id -> symbol lookup from entities DataFrame."""
    return dict(zip(
        entities["entity_id"].tolist(),
        entities["symbol"].tolist(),
        strict=True,
    ))


def _join_financial_data(
    tables: dict[str, pd.DataFrame],
    table_specs: tuple[TableSpec, ...],
    entity_symbols: dict[int, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Join financial tables with diagnostics on entity loss.

    The first table spec is the base. Subsequent tables are joined per
    their TableSpec.join_type. For each inner join, entities lost from
    the base are logged with reason "missing_from:{table_name}".

    Args:
        tables: Mapping of table_name -> loaded DataFrame.
        table_specs: Table specifications (defines join order and type).
        entity_symbols: entity_id -> symbol for logging.

    Returns:
        Tuple of (joined_df, dropped_records).
    """
    join_keys = ["entity_id", "fiscal_year", "period"]
    dropped_records: list[dict[str, Any]] = []

    base_spec = table_specs[0]
    result = tables[base_spec.table_name]

    for spec in table_specs[1:]:
        df = tables[spec.table_name]

        # Diagnostic: compare entity sets before join.
        base_entities = set(result["entity_id"].unique())
        join_entities = set(df["entity_id"].unique())

        if spec.join_type == "inner":
            lost = base_entities - join_entities
            if lost:
                logger.warning(
                    "%d entities in base missing from %s (will be dropped)",
                    len(lost),
                    spec.table_name,
                )
                for eid in lost:
                    dropped_records.append({
                        "symbol": entity_symbols.get(eid, "unknown"),
                        "company_name": "",
                        "reason": f"missing_from:{spec.table_name}",
                        "periods_present": 0,
                        "periods_required": 0,
                        "missing_fields": "",
                    })

        # Drop 'date' from joining table to avoid column collisions.
        right = df.drop(columns=["date"], errors="ignore")
        how: Literal["inner", "left"] = (
            "inner" if spec.join_type == "inner" else "left"
        )
        result = result.merge(
            right,
            on=join_keys,
            how=how,
            suffixes=("", f"_{spec.table_name}"),
        )

    return result, dropped_records


def _check_duplicates(df: pd.DataFrame) -> None:
    """Verify no duplicate (entity_id, fiscal_year, period) rows.

    Raises:
        RuntimeError: If duplicates exist.
    """
    dupes = df.duplicated(subset=["entity_id", "fiscal_year", "period"], keep=False)
    if dupes.any():
        dupe_count = dupes.sum()
        sample = df.loc[dupes, ["entity_id", "fiscal_year", "period"]].head(5)
        raise RuntimeError(
            f"{dupe_count} duplicate (entity_id, fiscal_year, period) rows "
            f"found. Sample:\n{sample}"
        )


# ---------------------------------------------------------------------------
# Common time period and company filtering
# ---------------------------------------------------------------------------


def _determine_common_range(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[int, int]:
    """Determine the common fiscal year range across all companies.

    Uses 10th/90th percentile of per-company min/max fiscal years to
    find a range where the majority of companies have data.

    Args:
        df: Joined financial data with entity_id and fiscal_year columns.
        config: Pipeline configuration.

    Returns:
        Tuple of (common_min_year, common_max_year).
    """
    company_ranges = df.groupby("entity_id")["fiscal_year"].agg(["min", "max"])

    if company_ranges.empty:
        raise RuntimeError(
            "No financial data after joining tables. Check that "
            "period_type, fiscal year range, and exchange filters "
            "match available data in the database."
        )

    common_min = int(company_ranges["min"].quantile(config.common_range_lower_pct))
    common_max = int(company_ranges["max"].quantile(config.common_range_upper_pct))

    # Clamp to configured bounds.
    common_min = max(common_min, config.min_fiscal_year)
    common_max = min(common_max, config.max_fiscal_year)

    logger.info(
        "Common fiscal year range: %d-%d (from %d companies)",
        common_min,
        common_max,
        len(company_ranges),
    )
    return common_min, common_max


def _filter_companies(
    df: pd.DataFrame,
    common_min: int,
    common_max: int,
    required_columns: set[str],
    entity_symbols: dict[int, str],
    period_type: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Drop companies with gaps, early endings, or NaN in required columns.

    Continuity-from-entry check:
    - Find each company's first fiscal year within [common_min, common_max].
    - Require data through common_max.
    - Require continuous fiscal years from first year to common_max.

    NaN handling driven by ColumnSpec.required:
    - required=True: any period with NaN -> drop company.
    - required=False: NaN passes through to downstream.

    Args:
        df: Joined financial data.
        common_min: Start of the common fiscal year range.
        common_max: End of the common fiscal year range.
        required_columns: Internal names of columns where NaN triggers drop.
        entity_symbols: entity_id -> symbol for logging.
        period_type: "FY" or "FQ".

    Returns:
        Tuple of (retained_df, dropped_records).
    """
    dropped_records: list[dict[str, Any]] = []

    # Filter to common range first.
    df_ranged = df[
        (df["fiscal_year"] >= common_min) & (df["fiscal_year"] <= common_max)
    ].copy()

    company_groups = df_ranged.groupby("entity_id")
    companies_to_keep: list[int] = []

    # Only check required columns that actually exist in the DataFrame.
    check_cols = sorted(required_columns & set(df_ranged.columns))

    for entity_id_raw, group in company_groups:
        entity_id = int(entity_id_raw)  # type: ignore[arg-type]
        symbol = entity_symbols.get(entity_id, "unknown")
        company_name = (
            str(group["company_name"].iloc[0])
            if "company_name" in group.columns
            else ""
        )

        # Year-level coverage (works for both FY and FQ).
        present_years = sorted(group["fiscal_year"].unique())
        first_year = present_years[0]
        last_year = present_years[-1]

        # Must have data through common_max.
        if last_year < common_max:
            dropped_records.append({
                "symbol": symbol,
                "company_name": company_name,
                "reason": "ends_before_common_max",
                "periods_present": len(present_years),
                "periods_required": common_max - first_year + 1,
                "missing_fields": "",
            })
            continue

        # Check for gaps from first_year to common_max.
        expected_years = set(range(first_year, common_max + 1))
        missing_years = sorted(expected_years - set(present_years))
        if missing_years:
            dropped_records.append({
                "symbol": symbol,
                "company_name": company_name,
                "reason": "temporal_gap",
                "periods_present": len(present_years),
                "periods_required": common_max - first_year + 1,
                "missing_fields": ",".join(str(y) for y in missing_years),
            })
            continue

        # Check required columns for ANY NaN.
        nan_fields: list[str] = []
        for col in check_cols:
            if group[col].isna().any():
                nan_years = group.loc[
                    group[col].isna(), "fiscal_year"
                ].tolist()
                nan_fields.append(col)
                logger.debug(
                    "Dropping %s: NaN in required column '%s' for years %s",
                    symbol,
                    col,
                    nan_years,
                )

        if nan_fields:
            dropped_records.append({
                "symbol": symbol,
                "company_name": company_name,
                "reason": "partial_nan:" + ",".join(nan_fields),
                "periods_present": len(present_years),
                "periods_required": common_max - first_year + 1,
                "missing_fields": ",".join(nan_fields),
            })
            continue

        companies_to_keep.append(entity_id)

    retained = df_ranged[df_ranged["entity_id"].isin(companies_to_keep)].copy()

    logger.info(
        "Retained %d companies, dropped %d",
        len(companies_to_keep),
        len(dropped_records),
    )
    return retained, dropped_records


# ---------------------------------------------------------------------------
# Currency filtering
# ---------------------------------------------------------------------------


def _filter_by_currency(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    config: AnalysisConfig,
    entity_symbols: dict[int, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter entities by reporting currency and cross-validate sources.

    Two-part approach:
    1. Cross-validate: compare companyProfile.currency against
       incomeStatement.reportedCurrency, log mismatches as warnings.
    2. Filter: drop entities whose reportedCurrency is not in
       config.currencies.

    Skips entirely if config.currencies is empty.

    Args:
        conn: Database connection.
        df: Joined financial data with entity_id column.
        config: Pipeline configuration.
        entity_symbols: entity_id -> symbol for logging.

    Returns:
        Tuple of (filtered_df, dropped_records).
    """
    if not config.currencies:
        logger.info("Currency filtering disabled (empty currencies tuple)")
        return df, []

    entity_ids = sorted(df["entity_id"].unique().tolist())
    if not entity_ids:
        return df, []

    placeholders = ", ".join("?" for _ in entity_ids)

    # Query reportedCurrency from incomeStatement.
    _validate_columns(conn, "incomeStatement", ["entityId", "reportedCurrency"])
    is_query = f"""
        SELECT DISTINCT entityId, reportedCurrency
        FROM incomeStatement
        WHERE entityId IN ({placeholders})
    """
    is_currencies = pd.read_sql_query(is_query, conn, params=entity_ids)
    is_currencies = is_currencies.rename(columns={
        "entityId": "entity_id",
        "reportedCurrency": "reported_currency",
    })

    # Query currency from companyProfile.
    _validate_columns(conn, "companyProfile", ["entityId", "currency"])
    cp_query = f"""
        SELECT entityId, currency
        FROM companyProfile
        WHERE entityId IN ({placeholders})
    """
    cp_currencies = pd.read_sql_query(cp_query, conn, params=entity_ids)
    cp_currencies = cp_currencies.rename(columns={
        "entityId": "entity_id",
        "currency": "profile_currency",
    })

    # Cross-validate: compare the two sources.
    for _, row in cp_currencies.iterrows():
        eid = int(row["entity_id"])
        profile_cur = row["profile_currency"]
        symbol = entity_symbols.get(eid, "unknown")

        is_rows = is_currencies[is_currencies["entity_id"] == eid]
        if is_rows.empty:
            continue

        is_curs = set(is_rows["reported_currency"].dropna().unique())
        if profile_cur and is_curs and profile_cur not in is_curs:
            logger.warning(
                "Currency mismatch for %s (entity %d): "
                "companyProfile.currency=%s, "
                "incomeStatement.reportedCurrency=%s",
                symbol,
                eid,
                profile_cur,
                sorted(is_curs),
            )

    # Filter: drop entities not in allowed currencies.
    # Use the most common reportedCurrency per entity from incomeStatement.
    entity_currency = (
        is_currencies
        .groupby("entity_id")["reported_currency"]
        .agg(lambda x: str(x.mode().iloc[0]) if not x.mode().empty else "")
        .reset_index()
    )

    allowed = set(config.currencies)
    dropped_records: list[dict[str, Any]] = []
    drop_entity_ids: list[int] = []

    for _, row in entity_currency.iterrows():
        eid = int(row["entity_id"])
        cur = row["reported_currency"]
        if cur not in allowed:
            symbol = entity_symbols.get(eid, "unknown")
            drop_entity_ids.append(eid)
            dropped_records.append({
                "symbol": symbol,
                "company_name": "",
                "reason": f"non_allowed_currency:{cur}",
                "periods_present": 0,
                "periods_required": 0,
                "missing_fields": "",
            })
            logger.info(
                "Dropping %s (entity %d): reportedCurrency=%s "
                "not in allowed currencies %s",
                symbol,
                eid,
                cur,
                sorted(allowed),
            )

    if drop_entity_ids:
        before_count = df["entity_id"].nunique()
        df = df[~df["entity_id"].isin(drop_entity_ids)].copy()
        after_count = df["entity_id"].nunique()
        logger.info(
            "Currency filter: dropped %d entities (%d -> %d)",
            before_count - after_count,
            before_count,
            after_count,
        )
    else:
        logger.info("Currency filter: all entities use allowed currencies")

    return df, dropped_records


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess(
    df: pd.DataFrame,
    required_columns: set[str],
    optional_columns: set[str],
) -> pd.DataFrame:
    """Apply preprocessing with differentiated NaN handling.

    - All numeric columns: inf -> NaN (inf is always bad data).
    - Required columns: NaN -> 0 (safety net after filtering).
    - Optional columns: NaN preserved for downstream.

    Args:
        df: Financial data DataFrame.
        required_columns: Internal names where NaN -> 0.
        optional_columns: Internal names where NaN is preserved.

    Returns:
        Preprocessed DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Step 1: inf -> NaN on ALL numeric columns.
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        logger.warning("Replacing %d inf values with NaN", inf_count)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Step 2: Verify no NaN remains in required columns after filtering.
    required_numeric = [c for c in numeric_cols if c in required_columns]
    if required_numeric:
        nan_count = df[required_numeric].isna().sum().sum()
        if nan_count > 0:
            nan_detail = df[required_numeric].isna().sum()
            affected = {col: int(n) for col, n in nan_detail.items() if n > 0}
            raise RuntimeError(
                f"{nan_count} NaN values in required columns after filtering "
                f"(indicates a filtering bug): {affected}"
            )

    # Log NaN counts in optional columns (informational, not replaced).
    optional_numeric = [c for c in numeric_cols if c in optional_columns]
    if optional_numeric:
        opt_nan = df[optional_numeric].isna().sum()
        for col_name, count in opt_nan.items():
            if count > 0:
                logger.info(
                    "Optional column '%s' has %d NaN values (preserved)",
                    col_name,
                    count,
                )

    return df


def _assign_period_idx(
    df: pd.DataFrame,
    common_min: int,
    period_type: str,
) -> pd.DataFrame:
    """Assign global calendar-based period_idx.

    FY: period_idx = fiscal_year - common_min
    FQ: period_idx = (fiscal_year - common_min) * 4 + quarter_offset
        where Q1=0, Q2=1, Q3=2, Q4=3.

    Args:
        df: Financial data.
        common_min: The base fiscal year (period_idx 0 = this year).
        period_type: "FY" or "FQ".

    Returns:
        DataFrame with period_idx column added, sorted by entity_id
        and period_idx.

    Raises:
        ValueError: If period_type is "FQ" and an unmapped quarter value
            is encountered.
    """
    df = df.sort_values(["entity_id", "fiscal_year"]).copy()

    if period_type == "FY":
        df["period_idx"] = df["fiscal_year"] - common_min
    else:
        if "period" not in df.columns:
            raise ValueError(
                "FQ mode requires a 'period' column in the DataFrame."
            )
        quarter_map = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
        quarters = df["period"].map(quarter_map)
        unmapped = quarters.isna()
        if unmapped.any():
            bad_values = df.loc[unmapped, "period"].unique().tolist()
            raise ValueError(
                f"Unmapped quarter values in period column: {bad_values}. "
                f"Expected one of {list(quarter_map.keys())}."
            )
        df["period_idx"] = (
            (df["fiscal_year"] - common_min) * 4 + quarters.astype(int)
        )
        df = df.sort_values(["entity_id", "period_idx"])

    return df


# ---------------------------------------------------------------------------
# Dropped companies log
# ---------------------------------------------------------------------------


def _write_dropped_log(
    dropped_records: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write dropped companies to a CSV log file.

    File columns: symbol, company_name, reason, periods_present,
    periods_required, missing_fields.

    Args:
        dropped_records: List of dropped company records.
        output_dir: Directory for the output file.

    Returns:
        Path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"dropped_companies_{timestamp}.csv"

    columns = [
        "symbol", "company_name", "reason",
        "periods_present", "periods_required", "missing_fields",
    ]

    if dropped_records:
        pd.DataFrame(dropped_records, columns=columns).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(path, index=False)

    logger.info(
        "Dropped companies log written to %s (%d entries)",
        path,
        len(dropped_records),
    )
    return path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def load_raw_data(
    config: AnalysisConfig,
    output_dir: Path | None = None,
) -> RawFinancialData:
    """Load and validate raw financial data from the FMP database.

    Single public entry point. Connects to the FMP SQLite database, loads
    all required fields per config.table_specs, validates completeness,
    drops companies with insufficient data or NaN in required columns,
    and returns a RawFinancialData contract object.

    Args:
        config: Pipeline configuration specifying database path, exchange
            filters, period type, fiscal year range, and column specs.
        output_dir: Directory for the dropped-companies log. Defaults to
            config.output_directory / "data_loading".

    Returns:
        RawFinancialData with validated, preprocessed financial data.

    Raises:
        FileNotFoundError: If the database file does not exist.
        RuntimeError: If required columns are missing from database tables,
            or if duplicate rows are detected.
    """
    if output_dir is None:
        output_dir = config.output_directory / "data_loading"

    # Validate all SQL identifiers from config before any queries.
    for spec in config.table_specs:
        _validate_identifier(spec.table_name, "table_specs.table_name")
        for col in spec.columns:
            _validate_identifier(col.db_column, f"{spec.table_name}.db_column")
    for col in config.entity_columns:
        _validate_identifier(col.db_column, "entity_columns.db_column")
    _validate_identifier(
        config.price_column.db_column, "price_column.db_column",
    )
    _validate_identifier(
        config.price_column.internal_name, "price_column.internal_name",
    )

    logger.info(
        "Loading data from %s (exchanges=%s, period=%s, years=%d-%d)",
        config.db_path,
        config.exchanges,
        config.period_type,
        config.min_fiscal_year,
        config.max_fiscal_year,
    )

    # Derive column requirements from config.
    required_columns, optional_columns = _get_column_requirements(config)
    all_dropped: list[dict[str, Any]] = []

    conn = _connect_readonly(config.db_path)
    try:
        # Step 1: Load eligible entities.
        entities = _load_eligible_entities(conn, config)
        entity_ids = entities["entity_id"].tolist()

        if not entity_ids:
            raise RuntimeError(
                f"No eligible entities found for exchanges {config.exchanges}"
            )

        entity_symbols = _build_entity_symbol_map(entities)

        # Step 2: Load financial tables (config-driven).
        tables: dict[str, pd.DataFrame] = {}
        for spec in config.table_specs:
            tables[spec.table_name] = _load_table_data(
                conn, spec, entity_ids, config,
            )
            logger.info(
                "Loaded %d rows from %s",
                len(tables[spec.table_name]),
                spec.table_name,
            )

        # Step 3: Join tables with diagnostics.
        joined, join_dropped = _join_financial_data(
            tables, config.table_specs, entity_symbols,
        )
        all_dropped.extend(join_dropped)
        logger.info("Joined data: %d rows", len(joined))

        _check_duplicates(joined)

        # Step 4: Filter by reporting currency.
        joined, currency_dropped = _filter_by_currency(
            conn, joined, config, entity_symbols,
        )
        all_dropped.extend(currency_dropped)

        # Step 5: Merge entity metadata.
        joined = joined.merge(entities, on="entity_id", how="inner")

        # Step 6: Align prices to fiscal period end dates.
        prices = _align_prices(conn, joined[["entity_id", "date"]], config)
        joined = joined.merge(prices, on=["entity_id", "date"], how="left")

    finally:
        conn.close()

    # Step 7: Common time period and company filtering.
    common_min, common_max = _determine_common_range(joined, config)
    retained, filter_dropped = _filter_companies(
        joined, common_min, common_max, required_columns, entity_symbols,
        period_type=config.period_type,
    )
    all_dropped.extend(filter_dropped)

    if retained.empty:
        raise RuntimeError(
            "No companies retained after filtering. Check data completeness "
            "and NaN rates in required columns."
        )

    # Step 8: Preprocessing (inf -> NaN -> 0 for required; inf -> NaN for optional).
    retained = _preprocess(retained, required_columns, optional_columns)

    # Step 9: Assign global calendar-based period_idx.
    retained = _assign_period_idx(
        retained, common_min=common_min, period_type=config.period_type,
    )

    # Step 10: Write dropped companies log.
    dropped_path = _write_dropped_log(all_dropped, output_dir)

    # Step 11: Build and return contract object.
    period_range = (
        int(retained["fiscal_year"].min()),
        int(retained["fiscal_year"].max()),
    )

    result = RawFinancialData(
        data=retained.reset_index(drop=True),
        query_metadata={
            "db_path": str(config.db_path),
            "exchanges": list(config.exchanges),
            "period_type": config.period_type,
            "min_fiscal_year": config.min_fiscal_year,
            "max_fiscal_year": config.max_fiscal_year,
            "entity_filter": {
                "isEtf": 0,
                "isAdr": 0,
                "isFund": 0,
                "isActivelyTrading": 1,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        },
        row_count=len(retained),
        company_count=retained["entity_id"].nunique(),
        period_range=period_range,
        dropped_companies_path=dropped_path,
    )

    logger.info(
        "Load complete: %d rows, %d companies, periods %d-%d",
        result.row_count,
        result.company_count,
        period_range[0],
        period_range[1],
    )
    return result
