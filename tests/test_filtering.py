"""Tests for pipeline.analysis.filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.analysis.filtering import apply_filters
from pipeline.config.settings import AnalysisConfig


def _make_companies(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    """Build a per-company DataFrame indexed by entity_id.

    Default: 5 companies, all passing default filters.
    """
    if rows is None:
        rows = [
            {"entity_id": i, "symbol": f"CO{i}", "fcf": 100.0 + i * 10,
             "operating_income": 50.0, "revenue": 200.0,
             "market_cap": 50_000_000.0, "debt_cash_ratio": 1.0}
            for i in range(1, 6)
        ]
    return pd.DataFrame(rows).set_index("entity_id")


# ---------------------------------------------------------------------------
# Individual filter toggle tests
# ---------------------------------------------------------------------------


class TestNegativeFcfFilter:
    """Filter 1: negative/zero FCF."""

    def test_removes_negative_fcf(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "GOOD", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
            {"entity_id": 2, "symbol": "BAD", "fcf": -10.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(enable_negative_fcf_filter=True)
        result, log = apply_filters(df, config)

        assert "GOOD" in result["symbol"].values
        assert "BAD" not in result["symbol"].values
        assert "BAD" in log.removed.get("negative_fcf", [])

    def test_removes_zero_fcf(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "ZERO", "fcf": 0.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(enable_negative_fcf_filter=True)
        result, _ = apply_filters(df, config)

        assert len(result) == 0

    def test_disabled_keeps_negative_fcf(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "NEG", "fcf": -10.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(enable_negative_fcf_filter=False)
        result, _ = apply_filters(df, config)

        assert len(result) == 1


class TestDataConsistencyFilter:
    """Filter 2: operating_income > revenue (when revenue > 0)."""

    def test_removes_oi_exceeding_revenue(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "OK", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
            {"entity_id": 2, "symbol": "ANOMALY", "fcf": 100.0,
             "operating_income": 300, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig()
        result, log = apply_filters(df, config)

        assert "ANOMALY" not in result["symbol"].values
        assert "ANOMALY" in log.removed.get("data_consistency", [])

    def test_zero_revenue_not_flagged(self) -> None:
        """When revenue = 0, OI > revenue should NOT trigger removal."""
        df = _make_companies([
            {"entity_id": 1, "symbol": "ZEROREV", "fcf": 100.0,
             "operating_income": 50, "revenue": 0.0, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig()
        result, _ = apply_filters(df, config)

        assert len(result) == 1

    def test_disabled(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "ANOMALY", "fcf": 100.0,
             "operating_income": 300, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(enable_data_consistency_filter=False)
        result, _ = apply_filters(df, config)

        assert len(result) == 1


class TestMarketCapFilter:
    """Filter 3: minimum market cap."""

    def test_removes_below_threshold(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "BIG", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
            {"entity_id": 2, "symbol": "SMALL", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 1e6,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig()  # min_market_cap = 20_000_000
        result, log = apply_filters(df, config)

        assert "SMALL" not in result["symbol"].values
        assert "SMALL" in log.removed.get("market_cap", [])

    def test_disabled(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "SMALL", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 1e6,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(enable_market_cap_filter=False)
        result, _ = apply_filters(df, config)

        assert len(result) == 1


class TestDebtCashFilter:
    """Filter 4: maximum debt-to-cash ratio."""

    def test_removes_high_ratio(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "OK", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.5},
            {"entity_id": 2, "symbol": "RISKY", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 5.0},
        ])
        config = AnalysisConfig()  # max_debt_to_cash_ratio = 2.5
        result, log = apply_filters(df, config)

        assert "RISKY" not in result["symbol"].values
        assert "RISKY" in log.removed.get("debt_cash", [])

    def test_inf_ratio_removed(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "NOCASH", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": float("inf")},
        ])
        config = AnalysisConfig()
        result, _ = apply_filters(df, config)

        assert len(result) == 0

    def test_disabled(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "RISKY", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 5.0},
        ])
        config = AnalysisConfig(enable_debt_cash_filter=False)
        result, _ = apply_filters(df, config)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# Owned-company bypass
# ---------------------------------------------------------------------------


class TestOwnedCompanyBypass:
    """Owned companies bypass filters in owned mode."""

    def test_owned_company_bypasses_filters(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "GOOD", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
            {"entity_id": 2, "symbol": "OWNED_BAD", "fcf": -10.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["OWNED_BAD"],
            enable_negative_fcf_filter=True,
        )
        result, log = apply_filters(df, config)

        # Owned company kept despite negative FCF.
        assert "OWNED_BAD" in result["symbol"].values
        assert "OWNED_BAD" in log.owned_bypassed

    def test_owned_tracking_records_failures(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "MY_CO", "fcf": -10.0,
             "operating_income": 50, "revenue": 200, "market_cap": 1e6,
             "debt_cash_ratio": 5.0},
        ])
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["MY_CO"],
            enable_negative_fcf_filter=True,
        )
        _, log = apply_filters(df, config)

        tracking = log.owned_tracking["MY_CO"]
        assert tracking["in_initial_data"] is True
        # Should fail negative_fcf, market_cap, and debt_cash filters.
        assert "negative_fcf" in tracking["filters_failed"]
        assert "market_cap" in tracking["filters_failed"]
        assert "debt_cash" in tracking["filters_failed"]

    def test_shortlist_mode_does_not_bypass(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "BAD", "fcf": -10.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(mode="shortlist", enable_negative_fcf_filter=True)
        result, _ = apply_filters(df, config)

        assert len(result) == 0

    def test_owned_not_in_data_tracked(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "OTHER", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["MISSING"],
        )
        _, log = apply_filters(df, config)

        assert log.owned_tracking["MISSING"]["in_initial_data"] is False


# ---------------------------------------------------------------------------
# Sort order and general behaviour
# ---------------------------------------------------------------------------


class TestSortOrder:
    """Post-filter sort by market_cap descending."""

    def test_sorted_by_market_cap_descending(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "SMALL", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 3e7,
             "debt_cash_ratio": 1.0},
            {"entity_id": 2, "symbol": "BIG", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 9e7,
             "debt_cash_ratio": 1.0},
            {"entity_id": 3, "symbol": "MED", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig()
        result, _ = apply_filters(df, config)

        caps = result["market_cap"].tolist()
        assert caps == sorted(caps, reverse=True)


class TestAllFiltersDisabled:
    """When all filters disabled, all companies pass."""

    def test_all_disabled_keeps_everything(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "BAD", "fcf": -10.0,
             "operating_income": 300, "revenue": 200, "market_cap": 1e6,
             "debt_cash_ratio": 10.0},
        ])
        config = AnalysisConfig(
            enable_negative_fcf_filter=False,
            enable_data_consistency_filter=False,
            enable_market_cap_filter=False,
            enable_debt_cash_filter=False,
        )
        result, _ = apply_filters(df, config)

        assert len(result) == 1


class TestMultiFilterReason:
    """FilterLog.reasons records first failing filter per symbol."""

    def test_reasons_records_first_filter(self) -> None:
        # Fails negative_fcf (first) AND market_cap (third).
        df = _make_companies([
            {"entity_id": 1, "symbol": "MULTI", "fcf": -10.0,
             "operating_income": 50, "revenue": 200, "market_cap": 1e6,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(enable_negative_fcf_filter=True)
        _, log = apply_filters(df, config)

        # First filter to catch it is negative_fcf.
        assert log.reasons["MULTI"] == "negative_fcf"
        # But it appears in both removed lists.
        assert "MULTI" in log.removed.get("negative_fcf", [])
        assert "MULTI" in log.removed.get("market_cap", [])


class TestEmptyInput:
    """Empty DataFrame input produces empty output without error."""

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=["symbol", "fcf", "operating_income", "revenue",
                     "market_cap", "debt_cash_ratio"],
        )
        df.index.name = "entity_id"
        config = AnalysisConfig()
        result, log = apply_filters(df, config)

        assert len(result) == 0
        assert log.removed == {}
        assert log.reasons == {}


class TestDisabledFilterOwnedTracking:
    """Disabled filter tracking only records for companies in data."""

    def test_disabled_filter_not_tracked_for_missing_owned(self) -> None:
        df = _make_companies([
            {"entity_id": 1, "symbol": "OTHER", "fcf": 100.0,
             "operating_income": 50, "revenue": 200, "market_cap": 5e7,
             "debt_cash_ratio": 1.0},
        ])
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["MISSING"],
            enable_negative_fcf_filter=False,
        )
        _, log = apply_filters(df, config)

        tracking = log.owned_tracking["MISSING"]
        assert tracking["in_initial_data"] is False
        # Disabled filter should NOT record "passed" for missing company.
        assert "negative_fcf" not in tracking["filters_passed"]


class TestValidation:
    """Input validation."""

    def test_missing_column_raises(self) -> None:
        df = _make_companies()
        df = df.drop(columns=["fcf"])

        with pytest.raises(ValueError, match="Missing required columns"):
            apply_filters(df, AnalysisConfig())
