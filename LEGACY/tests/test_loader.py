"""Tests for pipeline.data.loader."""

import logging
from pathlib import Path

import pytest

from pipeline.config.settings import AnalysisConfig
from pipeline.data.loader import load_raw_data

logger = logging.getLogger(__name__)

# The FMP database path. Skip tests if not available.
_DB_PATH = Path("/home/mattm/projects/Pers/financial_db/data/fmp.db")


@pytest.fixture
def au_config() -> AnalysisConfig:
    """Default AU market config for testing."""
    return AnalysisConfig(market="AU", period_type="FQ", history_years=5)


@pytest.fixture
def us_config() -> AnalysisConfig:
    """US market config for testing."""
    return AnalysisConfig(market="US", period_type="FQ", history_years=5)


@pytest.mark.skipif(not _DB_PATH.exists(), reason="FMP database not available")
class TestLoadRawData:
    def test_au_market_loads(self, au_config: AnalysisConfig, tmp_path: Path) -> None:
        result = load_raw_data(au_config, output_dir=tmp_path)

        assert result.row_count > 0
        assert result.company_count > 0
        assert result.period_range[0] <= result.period_range[1]
        assert result.dropped_companies_path.exists()

    def test_au_market_columns(self, au_config: AnalysisConfig, tmp_path: Path) -> None:
        result = load_raw_data(au_config, output_dir=tmp_path)
        df = result.data

        # Entity metadata columns.
        for col in ("entity_id", "symbol", "company_name", "exchange", "country"):
            assert col in df.columns, f"Missing entity column: {col}"

        # Financial columns from table specs.
        for col in ("revenue", "operating_income", "shares_diluted",
                     "lt_debt", "cash", "fcf"):
            assert col in df.columns, f"Missing financial column: {col}"

        # Growth columns (optional, may have NaN but column must exist).
        for col in ("fcf_growth", "revenue_growth"):
            assert col in df.columns, f"Missing growth column: {col}"

        # Price column.
        assert "adj_close" in df.columns

        # Period tracking.
        assert "fiscal_year" in df.columns
        assert "period_idx" in df.columns

    def test_au_market_no_nan_in_required(
        self, au_config: AnalysisConfig, tmp_path: Path,
    ) -> None:
        result = load_raw_data(au_config, output_dir=tmp_path)
        df = result.data

        required_cols = [
            "revenue", "operating_income", "shares_diluted",
            "lt_debt", "cash", "fcf", "adj_close",
        ]
        for col in required_cols:
            nan_count = df[col].isna().sum()
            assert nan_count == 0, (
                f"Required column '{col}' has {nan_count} NaN values"
            )

    def test_au_market_metadata(
        self, au_config: AnalysisConfig, tmp_path: Path,
    ) -> None:
        result = load_raw_data(au_config, output_dir=tmp_path)
        meta = result.query_metadata

        assert meta["exchanges"] == ["ASX"]
        assert meta["period_type"] == "FQ"

    def test_period_idx_monotonic(
        self, au_config: AnalysisConfig, tmp_path: Path,
    ) -> None:
        """period_idx should be monotonically increasing within each company."""
        result = load_raw_data(au_config, output_dir=tmp_path)
        df = result.data

        for entity_id, group in df.groupby("entity_id"):
            sorted_idx = group.sort_values("period_idx")["period_idx"]
            assert sorted_idx.is_monotonic_increasing, (
                f"entity_id {entity_id}: period_idx is not monotonically increasing"
            )

    def test_no_duplicate_rows(
        self, au_config: AnalysisConfig, tmp_path: Path,
    ) -> None:
        result = load_raw_data(au_config, output_dir=tmp_path)
        df = result.data

        dupes = df.duplicated(
            subset=["entity_id", "fiscal_year", "period"], keep=False,
        )
        assert not dupes.any(), (
            f"Found {dupes.sum()} duplicate rows"
        )
