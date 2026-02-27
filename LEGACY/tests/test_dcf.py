"""Tests for pipeline.analysis.dcf."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pipeline.analysis.dcf import calculate_all_dcf, calculate_dcf
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import IntrinsicValue, Projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config(**overrides: Any) -> AnalysisConfig:
    kwargs: dict[str, Any] = {
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.01,
        "margin_of_safety": 0.50,
        "quarters_per_year": 4,
        "projection_periods": (5,),
        "primary_period": 5,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _make_projection(
    quarterly_values: list[float] | None = None,
    quarterly_growth_rates: list[float] | None = None,
    period_years: int = 5,
    scenario: str = "base",
    annual_cagr: float = 0.05,
    current_value: float = 100.0,
) -> Projection:
    n = period_years * 4
    if quarterly_values is None:
        # Constant growth at 2% per quarter
        qv: list[float] = []
        v = current_value
        for _ in range(n):
            v = v * 1.02
            qv.append(v)
        quarterly_values = qv
    if quarterly_growth_rates is None:
        quarterly_growth_rates = [0.02] * n
    return Projection(
        entity_id=1,
        metric="fcf",
        period_years=period_years,
        scenario=scenario,
        quarterly_growth_rates=quarterly_growth_rates,
        quarterly_values=quarterly_values,
        annual_cagr=annual_cagr,
        current_value=current_value,
    )


# ---------------------------------------------------------------------------
# calculate_dcf — core
# ---------------------------------------------------------------------------

class TestCalculateDcf:

    def test_returns_intrinsic_value(self) -> None:
        config = _default_config()
        proj = _make_projection()
        result = calculate_dcf(100.0, 50.0, proj, config)
        assert isinstance(result, IntrinsicValue)

    def test_positive_fcf_gives_positive_iv(self) -> None:
        config = _default_config()
        proj = _make_projection(current_value=100.0)
        result = calculate_dcf(100.0, 50.0, proj, config)
        assert result.iv_per_share > 0
        assert result.present_value > 0
        assert result.terminal_value > 0

    def test_margin_of_safety_applied(self) -> None:
        """IV with 50% MoS should be half of IV with 0% MoS (conceptually)."""
        proj = _make_projection()
        config_50 = _default_config(margin_of_safety=0.50)
        config_01 = _default_config(margin_of_safety=0.01)

        iv_50 = calculate_dcf(100.0, 50.0, proj, config_50)
        iv_01 = calculate_dcf(100.0, 50.0, proj, config_01)

        # Same present value, different margin
        assert iv_50.present_value == pytest.approx(iv_01.present_value, rel=1e-6)
        # IV with 50% MoS ≈ half of IV with ~0% MoS
        ratio = iv_50.iv_per_share / iv_01.iv_per_share
        assert ratio == pytest.approx(0.50 / 0.99, rel=1e-2)

    def test_more_shares_reduces_iv_per_share(self) -> None:
        config = _default_config()
        proj = _make_projection()
        iv_few = calculate_dcf(100.0, 50.0, proj, config)
        iv_many = calculate_dcf(100.0, 100.0, proj, config)
        assert iv_many.iv_per_share < iv_few.iv_per_share

    def test_higher_discount_rate_reduces_pv(self) -> None:
        proj = _make_projection()
        config_low = _default_config(discount_rate=0.08)
        config_high = _default_config(discount_rate=0.15)
        iv_low = calculate_dcf(100.0, 50.0, proj, config_low)
        iv_high = calculate_dcf(100.0, 50.0, proj, config_high)
        assert iv_high.present_value < iv_low.present_value

    def test_scenario_and_period_preserved(self) -> None:
        config = _default_config()
        proj = _make_projection(period_years=5, scenario="optimistic")
        result = calculate_dcf(100.0, 50.0, proj, config)
        assert result.scenario == "optimistic"
        assert result.period_years == 5

    def test_config_rates_preserved(self) -> None:
        config = _default_config(
            discount_rate=0.12, terminal_growth_rate=0.02, margin_of_safety=0.40,
        )
        proj = _make_projection()
        result = calculate_dcf(100.0, 50.0, proj, config)
        assert result.discount_rate == 0.12
        assert result.terminal_growth_rate == 0.02
        assert result.margin_of_safety == 0.40

    def test_annual_cash_flows_correct_count(self) -> None:
        config = _default_config()
        proj = _make_projection(period_years=5)
        result = calculate_dcf(100.0, 50.0, proj, config)
        assert len(result.projected_annual_cash_flows) == 5

    def test_annual_cash_flows_sum_to_quarterly(self) -> None:
        """Each annual CF should be the sum of 4 quarterly CFs."""
        config = _default_config()
        proj = _make_projection(period_years=5)
        result = calculate_dcf(100.0, 50.0, proj, config)
        for y in range(5):
            start = y * 4
            end = start + 4
            quarterly_sum = sum(proj.quarterly_values[start:end])
            assert result.projected_annual_cash_flows[y] == pytest.approx(
                quarterly_sum, rel=1e-10,
            )


# ---------------------------------------------------------------------------
# calculate_dcf — hand calculation
# ---------------------------------------------------------------------------

class TestDcfHandCalculation:
    """Verify against a manually computed example."""

    def test_known_values(self) -> None:
        """Flat $100 quarterly FCF for 2 years (8 quarters).

        quarterly_discount = (1.10)^0.25 - 1 ≈ 0.024114
        PV of CFs = sum(100 / (1.024114)^q for q in 1..8) ≈ 750.52
        terminal: final_annual = 100 * 4 = 400
                  terminal_cf = 400 * 1.01 = 404
                  terminal_value = 404 / (0.10 - 0.01) = 4488.89
                  terminal_pv = 4488.89 / (1.10)^2 ≈ 3710.65
        PV = 750.52 + 3710.65 ≈ 4461.17
        IV/share = 4461.17 * 0.50 / 50 ≈ 44.61
        """
        config = _default_config(
            discount_rate=0.10,
            terminal_growth_rate=0.01,
            margin_of_safety=0.50,
        )
        flat_values = [100.0] * 8
        proj = _make_projection(
            quarterly_values=flat_values,
            quarterly_growth_rates=[0.0] * 8,
            period_years=2,
            current_value=100.0,
        )
        result = calculate_dcf(100.0, 50.0, proj, config)

        # PV of quarterly cash flows
        qd = (1.10) ** 0.25 - 1
        expected_pv_cfs = sum(100.0 / (1 + qd) ** q for q in range(1, 9))

        # Terminal value
        expected_terminal = 400.0 * 1.01 / (0.10 - 0.01)
        expected_terminal_pv = expected_terminal / (1.10**2)

        expected_pv = expected_pv_cfs + expected_terminal_pv
        expected_iv = expected_pv * 0.50 / 50.0

        assert result.present_value == pytest.approx(expected_pv, rel=1e-4)
        assert result.terminal_value == pytest.approx(expected_terminal, rel=1e-4)
        assert result.iv_per_share == pytest.approx(expected_iv, rel=1e-4)


# ---------------------------------------------------------------------------
# calculate_dcf — edge cases
# ---------------------------------------------------------------------------

class TestDcfEdgeCases:

    def test_negative_final_cf_gives_negative_terminal(self) -> None:
        """If projected FCF ends negative, terminal value is negative."""
        config = _default_config()
        proj = _make_projection(
            quarterly_values=[-10.0] * 20,
            quarterly_growth_rates=[0.0] * 20,
            period_years=5,
            current_value=-10.0,
        )
        result = calculate_dcf(-10.0, 50.0, proj, config)
        assert result.terminal_value < 0
        assert result.iv_per_share < 0

    def test_zero_fcf_gives_zero_iv(self) -> None:
        config = _default_config()
        proj = _make_projection(
            quarterly_values=[0.0] * 20,
            quarterly_growth_rates=[0.0] * 20,
            period_years=5,
            current_value=0.0,
        )
        result = calculate_dcf(0.0, 50.0, proj, config)
        assert result.present_value == pytest.approx(0.0)
        assert result.iv_per_share == pytest.approx(0.0)

    def test_growth_rate_from_projection(self) -> None:
        config = _default_config()
        proj = _make_projection(annual_cagr=0.12)
        result = calculate_dcf(100.0, 50.0, proj, config)
        assert result.growth_rate == 0.12

    def test_empty_quarterly_values_raises(self) -> None:
        config = _default_config()
        proj = _make_projection(
            quarterly_values=[],
            quarterly_growth_rates=[],
            period_years=5,
        )
        with pytest.raises(ValueError, match="empty quarterly_values"):
            calculate_dcf(100.0, 50.0, proj, config)


# ---------------------------------------------------------------------------
# calculate_all_dcf
# ---------------------------------------------------------------------------

class TestCalculateAllDcf:

    def _make_companies(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fcf": [100.0, 200.0],
                "shares_diluted": [50.0, 100.0],
                "revenue": [1000.0, 2000.0],
                "market_cap": [50e6, 100e6],
            },
            index=pd.Index([10, 20], name="entity_id"),
        )

    def _make_projections(self) -> dict[int, Any]:
        result = {}
        for eid, fcf in [(10, 100.0), (20, 200.0)]:
            proj = _make_projection(current_value=fcf)
            result[eid] = {
                5: {
                    "fcf": {
                        "base": proj,
                        "optimistic": _make_projection(
                            current_value=fcf, scenario="optimistic",
                        ),
                        "pessimistic": _make_projection(
                            current_value=fcf, scenario="pessimistic",
                        ),
                    },
                    "revenue": {
                        "base": _make_projection(current_value=1000.0),
                    },
                },
            }
        return result

    def test_all_companies_valued(self) -> None:
        config = _default_config()
        companies = self._make_companies()
        projections = self._make_projections()
        result = calculate_all_dcf(companies, projections, config)
        assert set(result.keys()) == {10, 20}

    def test_all_scenarios_valued(self) -> None:
        config = _default_config()
        companies = self._make_companies()
        projections = self._make_projections()
        result = calculate_all_dcf(companies, projections, config)
        for eid in (10, 20):
            assert set(result[eid][5].keys()) == {
                "base", "optimistic", "pessimistic",
            }

    def test_skips_zero_shares(self) -> None:
        config = _default_config()
        companies = pd.DataFrame(
            {
                "fcf": [100.0],
                "shares_diluted": [0.0],
                "revenue": [1000.0],
                "market_cap": [50e6],
            },
            index=pd.Index([10], name="entity_id"),
        )
        projections = {
            10: {5: {"fcf": {"base": _make_projection()}}},
        }
        result = calculate_all_dcf(companies, projections, config)
        assert 10 not in result

    def test_skips_missing_projections(self) -> None:
        config = _default_config()
        companies = pd.DataFrame(
            {
                "fcf": [100.0],
                "shares_diluted": [50.0],
                "revenue": [1000.0],
                "market_cap": [50e6],
            },
            index=pd.Index([10], name="entity_id"),
        )
        result = calculate_all_dcf(companies, {}, config)
        assert 10 not in result

    def test_empty_companies(self) -> None:
        config = _default_config()
        companies = pd.DataFrame(
            columns=["fcf", "shares_diluted"],
        )
        companies.index.name = "entity_id"
        result = calculate_all_dcf(companies, {}, config)
        assert result == {}

    def test_uses_only_fcf_not_revenue(self) -> None:
        """Revenue projections should not produce IntrinsicValues."""
        config = _default_config()
        companies = self._make_companies()
        projections = self._make_projections()
        result = calculate_all_dcf(companies, projections, config)
        # Result should not contain revenue-based IVs
        for eid in result:
            for period in result[eid]:
                for _scenario_name, iv in result[eid][period].items():
                    assert isinstance(iv, IntrinsicValue)
