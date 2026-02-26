"""Tests for pipeline.analysis.growth_stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis.growth_stats import (
    _calculate_cagr,
    _compute_raw_qoq_growth,
    compute_growth_statistics,
)
from pipeline.config.settings import AnalysisConfig


def _make_series(values: list[float]) -> pd.Series:
    """Build a Series indexed by sequential integers (like period_idx)."""
    return pd.Series(values, index=range(len(values)))


# ---------------------------------------------------------------------------
# CAGR unit tests
# ---------------------------------------------------------------------------


class TestCalculateCagr:
    """Legacy CAGR algorithm edge cases."""

    def test_normal_growth(self) -> None:
        # 100 -> 200 over 8 quarters: quarterly_cagr = 2^(1/8) - 1
        # annual_cagr = (1 + quarterly_cagr)^4 - 1 = 2^(4/8) - 1 ≈ 0.4142
        values = _make_series([100, 0, 0, 0, 0, 0, 0, 0, 200])
        cagr = _calculate_cagr(values)

        expected = 2 ** (4 / 8) - 1
        assert cagr == pytest.approx(expected, rel=1e-6)

    def test_fewer_than_two_valid_points(self) -> None:
        values = _make_series([100, np.nan, np.nan])
        assert _calculate_cagr(values) == 0.0

    def test_fewer_than_four_quarters_between(self) -> None:
        # 3 quarters between first and last valid: too short.
        values = _make_series([100, 110, 120, 130])
        assert _calculate_cagr(values) == 0.0

    def test_sign_change_returns_zero(self) -> None:
        values = _make_series([-100, 0, 0, 0, 0, 200])
        assert _calculate_cagr(values) == 0.0

    def test_both_negative_magnitude_increase(self) -> None:
        # Both negative, magnitude increased (more negative) -> negative CAGR.
        values = _make_series([-100, 0, 0, 0, 0, -200])
        cagr = _calculate_cagr(values)
        assert cagr < 0

    def test_both_negative_magnitude_decrease(self) -> None:
        # Both negative, magnitude decreased (less negative).
        # abs values shrunk (200->100), so quarterly CAGR is negative.
        # Legacy sign correction only flips when magnitude *increases*,
        # so this also yields negative CAGR.  Both-negative is a
        # degenerate case — CAGR is not meaningful on negative values.
        values = _make_series([-200, 0, 0, 0, 0, -100])
        cagr = _calculate_cagr(values)
        assert cagr < 0

    def test_zero_values_filtered_out(self) -> None:
        # Zeros are skipped; uses 100 (idx 0) and 200 (idx 8).
        values = _make_series([100, 0, 0, 0, 0, 0, 0, 0, 200])
        cagr = _calculate_cagr(values)
        assert cagr > 0

    def test_nan_values_filtered_out(self) -> None:
        values = _make_series([100, np.nan, np.nan, np.nan, np.nan, 200])
        cagr = _calculate_cagr(values)
        assert cagr > 0

    def test_all_zeros(self) -> None:
        values = _make_series([0, 0, 0, 0, 0])
        assert _calculate_cagr(values) == 0.0

    def test_all_nan(self) -> None:
        values = _make_series([np.nan, np.nan, np.nan])
        assert _calculate_cagr(values) == 0.0


# ---------------------------------------------------------------------------
# Raw QoQ growth unit tests
# ---------------------------------------------------------------------------


class TestComputeRawQoqGrowth:
    """Raw quarter-over-quarter growth computation."""

    def test_constant_quarterly_growth(self) -> None:
        """With constant quarterly growth g, every QoQ rate equals g."""
        g = 0.05
        values = pd.Series([100 * (1 + g) ** i for i in range(10)])
        qoq_growth = _compute_raw_qoq_growth(values)
        valid = qoq_growth.dropna()

        # 10 periods → 9 QoQ values (first is NaN from shift)
        assert len(valid) == 9
        for v in valid:
            assert v == pytest.approx(g, abs=1e-10)

    def test_two_periods_gives_one_observation(self) -> None:
        """Minimum input: 2 periods produces 1 QoQ rate."""
        values = pd.Series([100, 110])
        qoq_growth = _compute_raw_qoq_growth(values)
        valid = qoq_growth.dropna()
        assert len(valid) == 1
        assert valid.iloc[0] == pytest.approx(0.10, abs=1e-10)

    def test_single_period_gives_no_observations(self) -> None:
        values = pd.Series([100])
        qoq_growth = _compute_raw_qoq_growth(values)
        assert qoq_growth.dropna().empty

    def test_zero_denominator_produces_nan(self) -> None:
        """If previous quarter's value is zero, QoQ rate is NaN."""
        values = pd.Series([100, 0, 50])
        qoq_growth = _compute_raw_qoq_growth(values)
        # QoQ at index 1: (0 - 100) / |100| = -1.0 (valid)
        # QoQ at index 2: (50 - 0) / |0| = inf → NaN
        assert qoq_growth.iloc[1] == pytest.approx(-1.0)
        assert pd.isna(qoq_growth.iloc[2])

    def test_negative_values_improving(self) -> None:
        """FCF improving from negative toward zero: QoQ rates are positive."""
        values = pd.Series([-200, -180, -160, -140])
        qoq_growth = _compute_raw_qoq_growth(values)
        valid = qoq_growth.dropna()
        # g = (Q_t - Q_{t-1}) / |Q_{t-1}|
        # g_1 = (-180 - (-200)) / |-200| = 20/200 = 0.10
        assert all(v > 0 for v in valid)

    def test_sign_transition(self) -> None:
        """Absolute-value denominator handles sign transitions."""
        values = pd.Series([-100, -50, 50, 100])
        qoq_growth = _compute_raw_qoq_growth(values)
        valid = qoq_growth.dropna()
        # g_1 = (-50 - (-100)) / |-100| = 50/100 = 0.5
        # g_2 = (50 - (-50)) / |-50| = 100/50 = 2.0
        # g_3 = (100 - 50) / |50| = 50/50 = 1.0
        assert len(valid) == 3
        assert valid.iloc[0] == pytest.approx(0.5)
        assert valid.iloc[1] == pytest.approx(2.0)
        assert valid.iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_growth_statistics tests
# ---------------------------------------------------------------------------


def _make_growth_data(
    entity_id: int = 1,
    n_periods: int = 10,
    fcf_growth: float = 0.05,
    revenue_growth: float = 0.03,
    fcf_base: float = 100.0,
    revenue_base: float = 1000.0,
) -> pd.DataFrame:
    """Build a multi-period DataFrame for one company with constant growth.

    Generates absolute FCF and revenue values from a base value and
    constant quarterly growth rate. 10 periods (default) yields 9 QoQ
    growth observations, well above the minimum of 3.
    """
    rows = []
    fcf = fcf_base
    rev = revenue_base
    for i in range(n_periods):
        rows.append({
            "entity_id": entity_id,
            "period_idx": i,
            "fcf": fcf,
            "revenue": rev,
        })
        fcf *= 1 + fcf_growth
        rev *= 1 + revenue_growth
    return pd.DataFrame(rows)


class TestComputeGrowthStatistics:
    """Core functionality."""

    def test_mean_of_constant_growth(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data(fcf_growth=0.05, revenue_growth=0.03)
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_growth_mean"] == pytest.approx(0.05, abs=1e-9)
        assert result.loc[1, "revenue_growth_mean"] == pytest.approx(0.03, abs=1e-9)

    def test_variance_of_constant_growth_is_zero(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data(fcf_growth=0.05, revenue_growth=0.03)
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_growth_var"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc[1, "revenue_growth_var"] == pytest.approx(0.0, abs=1e-9)

    def test_std_of_constant_growth_is_zero(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data(fcf_growth=0.05, revenue_growth=0.03)
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_growth_std"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc[1, "revenue_growth_std"] == pytest.approx(0.0, abs=1e-9)

    def test_combined_growth_mean(self) -> None:
        config = AnalysisConfig()  # defaults: fcf=0.7, rev=0.3
        data = _make_growth_data(fcf_growth=0.10, revenue_growth=0.06)
        result = compute_growth_statistics(data, config)

        expected = 0.10 * 0.7 + 0.06 * 0.3
        assert result.loc[1, "combined_growth_mean"] == pytest.approx(expected, abs=1e-9)

    def test_growth_stability_constant_growth(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data(fcf_growth=0.05, revenue_growth=0.03)
        result = compute_growth_statistics(data, config)

        # std = 0.0 for both -> avg_std = 0 -> stability = 1/(1+0) = 1.0
        assert result.loc[1, "growth_stability"] == pytest.approx(1.0, abs=1e-9)

    def test_growth_stability_with_variance(self) -> None:
        """Variable FCF/revenue produces non-zero QoQ growth std."""
        config = AnalysisConfig()
        # 10 periods with alternating high/low quarters to create variance.
        fcf_values = [100, 120, 90, 130, 110, 140, 95, 135, 115, 145]
        rev_values = [1000, 1050, 980, 1070, 1040, 1100, 990, 1080, 1060, 1120]
        data = pd.DataFrame({
            "entity_id": [1] * 10,
            "period_idx": list(range(10)),
            "fcf": fcf_values,
            "revenue": rev_values,
        })
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_growth_std"] > 0  # type: ignore[operator]
        assert result.loc[1, "revenue_growth_std"] > 0  # type: ignore[operator]
        assert result.loc[1, "growth_stability"] < 1.0  # type: ignore[operator]

    def test_cagr_computed_from_absolute_values(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data(
            fcf_growth=0.05, revenue_growth=0.03, n_periods=10,
        )
        result = compute_growth_statistics(data, config)

        # CAGR should be positive and reflect quarterly compounding.
        assert result.loc[1, "fcf_cagr"] > 0  # type: ignore[operator]
        assert result.loc[1, "revenue_cagr"] > 0  # type: ignore[operator]


class TestMinDataPoints:
    """Behaviour with insufficient data for QoQ growth computation."""

    def test_few_periods_defaults_to_zero(self) -> None:
        """3 periods yields 2 QoQ growth points → below minimum of 3."""
        config = AnalysisConfig()
        data = _make_growth_data(n_periods=3)
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_growth_mean"] == 0.0
        assert result.loc[1, "fcf_growth_var"] == 0.0
        assert result.loc[1, "fcf_growth_std"] == 0.0

    def test_nan_absolute_values_reduce_qoq_count(self) -> None:
        """NaN in absolute values propagates to QoQ, reducing valid count."""
        config = AnalysisConfig()
        # 10 periods, but NaN in fcf at idx 2 and 5.
        # Each NaN kills two QoQ values (the NaN quarter and the next).
        fcf = [100.0] * 10
        fcf[2] = np.nan
        fcf[5] = np.nan
        data = pd.DataFrame({
            "entity_id": [1] * 10,
            "period_idx": list(range(10)),
            "fcf": fcf,
            "revenue": [1000 * 1.03**i for i in range(10)],
        })
        result = compute_growth_statistics(data, config)

        # Revenue is unaffected by FCF NaN.
        assert result.loc[1, "revenue_growth_mean"] != 0.0

    def test_very_short_series_defaults_to_zero(self) -> None:
        """2 periods: only 1 QoQ value → below minimum of 3."""
        config = AnalysisConfig()
        data = pd.DataFrame({
            "entity_id": [1, 1],
            "period_idx": [0, 1],
            "fcf": [100, 110],
            "revenue": [1000, 1060],
        })
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_growth_mean"] == 0.0
        assert result.loc[1, "fcf_growth_var"] == 0.0
        assert result.loc[1, "fcf_growth_std"] == 0.0


class TestMultipleCompanies:
    """Multiple companies in the same DataFrame."""

    def test_separate_stats_per_company(self) -> None:
        config = AnalysisConfig()
        data = pd.concat([
            _make_growth_data(entity_id=1, fcf_growth=0.05, revenue_growth=0.03),
            _make_growth_data(entity_id=2, fcf_growth=0.10, revenue_growth=0.08),
        ], ignore_index=True)

        result = compute_growth_statistics(data, config)

        assert len(result) == 2
        assert result.loc[1, "fcf_growth_mean"] == pytest.approx(0.05, abs=1e-9)
        assert result.loc[2, "fcf_growth_mean"] == pytest.approx(0.10, abs=1e-9)


class TestFcfReliability:
    """FCF reliability metric."""

    def test_all_positive_fcf(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data(fcf_base=100.0)
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_reliability"] == pytest.approx(1.0)

    def test_mixed_positive_negative_fcf(self) -> None:
        config = AnalysisConfig()
        data = pd.DataFrame({
            "entity_id": [1] * 10,
            "period_idx": list(range(10)),
            "fcf": [100, -50, 80, -30, 90, 100, 110, 120, 130, 140],
            "revenue": [1000 * 1.03**i for i in range(10)],
        })
        result = compute_growth_statistics(data, config)

        # 8 positive out of 10
        assert result.loc[1, "fcf_reliability"] == pytest.approx(0.8)

    def test_all_negative_fcf(self) -> None:
        config = AnalysisConfig()
        data = pd.DataFrame({
            "entity_id": [1] * 10,
            "period_idx": list(range(10)),
            "fcf": [-100 - i * 10 for i in range(10)],
            "revenue": [1000 * 1.03**i for i in range(10)],
        })
        result = compute_growth_statistics(data, config)

        assert result.loc[1, "fcf_reliability"] == pytest.approx(0.0)

    def test_nan_fcf_excluded_from_count(self) -> None:
        config = AnalysisConfig()
        # 8 values: 6 positive, 2 NaN. Reliability = 6/8 = 0.75
        fcf = [100.0, 200.0, np.nan, 300.0, 400.0, np.nan, 500.0, 600.0,
               700.0, 800.0]
        data = pd.DataFrame({
            "entity_id": [1] * 10,
            "period_idx": list(range(10)),
            "fcf": fcf,
            "revenue": [1000 * 1.03**i for i in range(10)],
        })
        result = compute_growth_statistics(data, config)

        # 8 non-NaN, all positive → 1.0
        assert result.loc[1, "fcf_reliability"] == pytest.approx(1.0)


class TestValidation:
    """Input validation."""

    def test_missing_column_raises(self) -> None:
        config = AnalysisConfig()
        data = _make_growth_data()
        data = data.drop(columns=["fcf"])

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_growth_statistics(data, config)
