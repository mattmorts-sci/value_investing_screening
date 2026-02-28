"""Tests for chart functions."""

from __future__ import annotations

import base64

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pipeline.config import DCFConfig, HeatmapConfig, SimulationConfig  # noqa: E402
from pipeline.output.charts import (  # noqa: E402
    fcf_projection_chart,
    figure_to_data_uri,
    iv_scenario_chart,
    margin_time_series_chart,
    revenue_growth_chart,
    revenue_projection_chart,
    sensitivity_heatmap_chart,
)
from pipeline.simulation import (  # noqa: E402
    PathData,
    PercentileBands,
    SimulationInput,
)

NUM_PROJ_QUARTERS = 8  # 2 years of projection


@pytest.fixture
def sample_paths() -> list[PathData]:
    """Two sample paths for testing."""
    rng = np.random.default_rng(42)
    paths = []
    for _ in range(2):
        rev = 100.0 * np.cumprod(1 + rng.normal(0.02, 0.05, NUM_PROJ_QUARTERS))
        fcf = rev * rng.uniform(0.05, 0.15, NUM_PROJ_QUARTERS)
        paths.append(PathData(quarterly_revenue=rev, quarterly_fcf=fcf))
    return paths


@pytest.fixture
def revenue_bands() -> PercentileBands:
    """Percentile bands for revenue."""
    base = 100.0 * np.cumprod(1 + np.full(NUM_PROJ_QUARTERS, 0.02))
    return PercentileBands(
        p10=base * 0.8,
        p25=base * 0.9,
        p50=base,
        p75=base * 1.1,
        p90=base * 1.2,
    )


@pytest.fixture
def fcf_bands() -> PercentileBands:
    """Percentile bands for FCF."""
    base = 10.0 * np.cumprod(1 + np.full(NUM_PROJ_QUARTERS, 0.02))
    return PercentileBands(
        p10=base * 0.5,
        p25=base * 0.7,
        p50=base,
        p75=base * 1.3,
        p90=base * 1.5,
    )


@pytest.fixture
def historical_revenues() -> list[float]:
    """12 quarters of historical revenue."""
    return [90.0 + i * 2.0 for i in range(12)]


@pytest.fixture
def historical_fcfs() -> list[float]:
    """12 quarters of historical FCF."""
    return [8.0 + i * 0.5 for i in range(12)]


@pytest.fixture
def sim_input() -> SimulationInput:
    """Simulation input for heatmap tests."""
    return SimulationInput(
        revenue_qoq_growth_mean=0.02,
        revenue_qoq_growth_var=0.001,
        margin_intercept=0.15,
        margin_slope=0.001,
        conversion_intercept=0.7,
        conversion_slope=0.0,
        conversion_median=None,
        conversion_is_fallback=False,
        starting_revenue=100.0,
        shares_outstanding=10.0,
        num_historical_quarters=12,
    )


# --- figure_to_data_uri ---


class TestFigureToDataUri:
    """Tests for figure_to_data_uri."""

    def test_returns_data_uri_prefix(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        uri = figure_to_data_uri(fig)
        assert uri.startswith("data:image/png;base64,")

    def test_contains_valid_png(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        uri = figure_to_data_uri(fig)
        b64_part = uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


# --- revenue_projection_chart ---


class TestRevenueProjectionChart:
    """Tests for revenue_projection_chart."""

    def test_returns_figure(
        self,
        sample_paths: list[PathData],
        revenue_bands: PercentileBands,
        historical_revenues: list[float],
    ) -> None:
        fig = revenue_projection_chart(
            sample_paths, revenue_bands, historical_revenues,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_title(
        self,
        sample_paths: list[PathData],
        revenue_bands: PercentileBands,
        historical_revenues: list[float],
    ) -> None:
        fig = revenue_projection_chart(
            sample_paths, revenue_bands, historical_revenues,
        )
        assert fig.axes[0].get_title() == "Revenue Projection"
        plt.close(fig)

    def test_no_historical(
        self,
        sample_paths: list[PathData],
        revenue_bands: PercentileBands,
    ) -> None:
        fig = revenue_projection_chart(sample_paths, revenue_bands, [])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_sample_paths(
        self,
        revenue_bands: PercentileBands,
        historical_revenues: list[float],
    ) -> None:
        """Bands still render with empty sample paths."""
        fig = revenue_projection_chart([], revenue_bands, historical_revenues)
        assert isinstance(fig, Figure)
        plt.close(fig)


# --- fcf_projection_chart ---


class TestFcfProjectionChart:
    """Tests for fcf_projection_chart."""

    def test_returns_figure(
        self,
        sample_paths: list[PathData],
        fcf_bands: PercentileBands,
        historical_fcfs: list[float],
    ) -> None:
        fig = fcf_projection_chart(
            sample_paths, fcf_bands, historical_fcfs,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_title(
        self,
        sample_paths: list[PathData],
        fcf_bands: PercentileBands,
        historical_fcfs: list[float],
    ) -> None:
        fig = fcf_projection_chart(
            sample_paths, fcf_bands, historical_fcfs,
        )
        assert fig.axes[0].get_title() == "Free Cash Flow Projection"
        plt.close(fig)


# --- Y-axis zero inclusion ---


class TestNegativeYAxisZero:
    """Y-axis includes zero when negative values are present."""

    def test_includes_zero_when_negative_bands(
        self,
        sample_paths: list[PathData],
        historical_revenues: list[float],
    ) -> None:
        base = np.array([100.0] * NUM_PROJ_QUARTERS)
        bands = PercentileBands(
            p10=base * -0.2,
            p25=base * 0.5,
            p50=base,
            p75=base * 1.5,
            p90=base * 2.0,
        )
        fig = revenue_projection_chart(
            sample_paths, bands, historical_revenues,
        )
        ymin, _ = fig.axes[0].get_ylim()
        assert ymin <= 0
        plt.close(fig)

    def test_includes_zero_when_negative_historical(
        self,
        sample_paths: list[PathData],
        revenue_bands: PercentileBands,
    ) -> None:
        hist = [-10.0, 5.0, 10.0, 15.0]
        fig = revenue_projection_chart(sample_paths, revenue_bands, hist)
        ymin, _ = fig.axes[0].get_ylim()
        assert ymin <= 0
        plt.close(fig)


# --- iv_scenario_chart ---


class TestIvScenarioChart:
    """Tests for iv_scenario_chart."""

    def test_returns_figure(self) -> None:
        fig = iv_scenario_chart(10.0, 15.0, 20.0, 12.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_title(self) -> None:
        fig = iv_scenario_chart(10.0, 15.0, 20.0, 12.0)
        assert fig.axes[0].get_title() == "Intrinsic Value Scenarios"
        plt.close(fig)

    def test_three_bars(self) -> None:
        fig = iv_scenario_chart(10.0, 15.0, 20.0, 12.0)
        ax = fig.axes[0]
        bars = list(ax.patches)
        assert len(bars) == 3
        plt.close(fig)

    def test_x_starts_at_zero(self) -> None:
        fig = iv_scenario_chart(10.0, 15.0, 20.0, 12.0)
        ax = fig.axes[0]
        xleft, _ = ax.get_xlim()
        assert xleft == 0
        plt.close(fig)


# --- sensitivity_heatmap_chart ---


class TestSensitivityHeatmapChart:
    """Tests for sensitivity_heatmap_chart."""

    @pytest.fixture
    def tiny_heatmap_config(self) -> HeatmapConfig:
        """2x2 grid with minimal replicates."""
        return HeatmapConfig(
            discount_rate_min=0.08,
            discount_rate_max=0.10,
            discount_rate_step=0.02,
            growth_multiplier_min=0.75,
            growth_multiplier_max=1.25,
            growth_multiplier_step=0.50,
            heatmap_replicates=50,
        )

    @pytest.fixture
    def tiny_sim_config(self) -> SimulationConfig:
        return SimulationConfig(num_replicates=50, num_display_paths=0)

    @pytest.fixture
    def short_dcf_config(self) -> DCFConfig:
        return DCFConfig(projection_years=2)

    def test_returns_figure(
        self,
        sim_input: SimulationInput,
        tiny_heatmap_config: HeatmapConfig,
        tiny_sim_config: SimulationConfig,
        short_dcf_config: DCFConfig,
    ) -> None:
        fig = sensitivity_heatmap_chart(
            sim_input=sim_input,
            current_price=50.0,
            heatmap_config=tiny_heatmap_config,
            sim_config=tiny_sim_config,
            dcf_config=short_dcf_config,
            seed=42,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_title(
        self,
        sim_input: SimulationInput,
        tiny_heatmap_config: HeatmapConfig,
        tiny_sim_config: SimulationConfig,
        short_dcf_config: DCFConfig,
    ) -> None:
        fig = sensitivity_heatmap_chart(
            sim_input=sim_input,
            current_price=50.0,
            heatmap_config=tiny_heatmap_config,
            sim_config=tiny_sim_config,
            dcf_config=short_dcf_config,
            seed=42,
        )
        assert fig.axes[0].get_title() == "Sensitivity Analysis"
        plt.close(fig)


# --- margin_time_series_chart ---


class TestMarginTimeSeriesChart:
    """Tests for margin_time_series_chart."""

    def test_returns_figure(self) -> None:
        fig = margin_time_series_chart(
            [100.0, 110.0, 120.0, 130.0],
            [15.0, 17.0, 19.0, 21.0],
            0.15, 0.005,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_title(self) -> None:
        fig = margin_time_series_chart(
            [100.0, 110.0], [15.0, 17.0], 0.15, 0.005,
        )
        assert fig.axes[0].get_title() == "Operating Margin"
        plt.close(fig)

    def test_handles_zero_revenue(self) -> None:
        """Quarters with zero revenue should not cause errors."""
        fig = margin_time_series_chart(
            [100.0, 0.0, 120.0, 130.0],
            [15.0, 0.0, 19.0, 21.0],
            0.15, 0.005,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# --- revenue_growth_chart ---


class TestRevenueGrowthChart:
    """Tests for revenue_growth_chart."""

    def test_returns_figure(self) -> None:
        revenues = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        fig = revenue_growth_chart(revenues)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_title(self) -> None:
        revenues = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        fig = revenue_growth_chart(revenues)
        assert fig.axes[0].get_title() == "Revenue Growth (YoY)"
        plt.close(fig)

    def test_insufficient_data_message(self) -> None:
        """Fewer than 5 quarters: shows 'Insufficient data'."""
        fig = revenue_growth_chart([100.0, 105.0, 110.0])
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert "Insufficient data" in texts
        plt.close(fig)

    def test_negative_growth(self) -> None:
        """Declining revenue produces bars without errors."""
        revenues = [100.0, 100.0, 100.0, 100.0, 80.0, 80.0, 80.0, 80.0]
        fig = revenue_growth_chart(revenues)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_mean_and_std_lines(self) -> None:
        """Mean and std lines are present when sufficient data."""
        revenues = [100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0]
        fig = revenue_growth_chart(revenues)
        ax = fig.axes[0]
        labels = [str(line.get_label()) for line in ax.get_lines()]
        has_mean = any("Mean" in lbl for lbl in labels)
        assert has_mean
        plt.close(fig)
