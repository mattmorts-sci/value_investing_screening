"""Tests for pipeline.analysis.derived_metrics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pipeline.analysis.derived_metrics import compute_derived_metrics


def _make_data(**overrides: object) -> pd.DataFrame:
    """Build a minimal multi-period DataFrame for one company."""
    defaults = {
        "entity_id": [1, 1, 1],
        "period_idx": [0, 1, 2],
        "symbol": ["AAA"] * 3,
        "company_name": ["Acme"] * 3,
        "exchange": ["ASX"] * 3,
        "country": ["AU"] * 3,
        "fcf": [100.0, 110.0, 120.0],
        "revenue": [1000.0, 1100.0, 1200.0],
        "operating_income": [200.0, 220.0, 240.0],
        "shares_diluted": [50.0, 50.0, 50.0],
        "lt_debt": [300.0, 300.0, 300.0],
        "cash": [100.0, 100.0, 100.0],
        "adj_close": [10.0, 11.0, 12.0],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestComputeDerivedMetrics:
    """Core functionality tests."""

    def test_selects_latest_period(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        assert len(result) == 1
        assert result.loc[1, "adj_close"] == 12.0

    def test_market_cap(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        # 12.0 * 50.0 = 600.0
        assert result.loc[1, "market_cap"] == 600.0

    def test_enterprise_value(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        # market_cap(600) + lt_debt(300) - cash(100) = 800
        assert result.loc[1, "enterprise_value"] == 800.0

    def test_debt_cash_ratio(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        # 300 / 100 = 3.0
        assert result.loc[1, "debt_cash_ratio"] == 3.0

    def test_fcf_per_share(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        # 120 / 50 = 2.4
        assert result.loc[1, "fcf_per_share"] == 2.4

    def test_acquirers_multiple(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        # EV(800) / OI(240) ≈ 3.333
        assert result.loc[1, "acquirers_multiple"] == pytest.approx(800 / 240)

    def test_fcf_to_market_cap(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        # 120 / 600 = 0.2
        assert result.loc[1, "fcf_to_market_cap"] == 0.2


class TestDivisionByZero:
    """Edge cases where denominators are zero."""

    def test_zero_cash_gives_inf_debt_cash_ratio(self) -> None:
        data = _make_data(cash=[0.0, 0.0, 0.0])
        result = compute_derived_metrics(data)

        assert math.isinf(result.loc[1, "debt_cash_ratio"])  # type: ignore[arg-type]

    def test_zero_operating_income_gives_inf_acquirers_multiple(self) -> None:
        data = _make_data(operating_income=[0.0, 0.0, 0.0])
        result = compute_derived_metrics(data)

        assert math.isinf(result.loc[1, "acquirers_multiple"])  # type: ignore[arg-type]

    def test_zero_market_cap_gives_inf_fcf_to_market_cap(self) -> None:
        # adj_close=0 -> market_cap=0
        data = _make_data(adj_close=[0.0, 0.0, 0.0])
        result = compute_derived_metrics(data)

        assert math.isinf(result.loc[1, "fcf_to_market_cap"])  # type: ignore[arg-type]


class TestMultipleCompanies:
    """Test with multiple companies in the DataFrame."""

    def test_one_row_per_company(self) -> None:
        data = pd.concat([
            _make_data(),
            _make_data(
                entity_id=[2, 2],
                period_idx=[0, 1],
                symbol=["BBB", "BBB"],
                company_name=["Beta", "Beta"],
                fcf=[50.0, 60.0],
                revenue=[500.0, 600.0],
                operating_income=[100.0, 120.0],
                shares_diluted=[25.0, 25.0],
                lt_debt=[150.0, 150.0],
                cash=[50.0, 50.0],
                adj_close=[5.0, 6.0],
                exchange=["ASX", "ASX"],
                country=["AU", "AU"],
            ),
        ], ignore_index=True)

        result = compute_derived_metrics(data)

        assert len(result) == 2
        assert set(result.index) == {1, 2}
        # Company 1 latest adj_close=12.0, company 2 latest adj_close=6.0
        assert result.loc[1, "adj_close"] == 12.0
        assert result.loc[2, "adj_close"] == 6.0


class TestMetadataCarriedForward:
    """Verify that entity metadata columns are preserved."""

    def test_symbol_preserved(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        assert result.loc[1, "symbol"] == "AAA"

    def test_company_name_preserved(self) -> None:
        data = _make_data()
        result = compute_derived_metrics(data)

        assert result.loc[1, "company_name"] == "Acme"


class TestValidation:
    """Input validation."""

    def test_missing_column_raises(self) -> None:
        data = _make_data()
        data = data.drop(columns=["adj_close"])

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_derived_metrics(data)
