"""Tests for pipeline.data.contracts."""

from pathlib import Path

import pandas as pd

from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import (
    AnalysisResults,
    FilterLog,
    IntrinsicValue,
    Projection,
    RawFinancialData,
)


class TestRawFinancialData:
    def test_construction(self) -> None:
        rfd = RawFinancialData(
            data=pd.DataFrame({"a": [1]}),
            query_metadata={"db": "test"},
            row_count=1,
            company_count=1,
            period_range=(2020, 2025),
            dropped_companies_path=Path("dropped.csv"),
        )
        assert rfd.row_count == 1
        assert rfd.period_range == (2020, 2025)


class TestFilterLog:
    def test_defaults(self) -> None:
        fl = FilterLog()
        assert fl.removed == {}
        assert fl.reasons == {}
        assert fl.owned_bypassed == []
        assert fl.owned_tracking == {}


class TestProjection:
    def test_construction(self) -> None:
        p = Projection(
            entity_id=1,
            metric="fcf",
            period_years=5,
            scenario="base",
            quarterly_growth_rates=[0.01, 0.02],
            quarterly_values=[100.0, 101.0],
            annual_cagr=0.05,
            current_value=100.0,
        )
        assert p.metric == "fcf"
        assert p.scenario == "base"


class TestIntrinsicValue:
    def test_construction(self) -> None:
        iv = IntrinsicValue(
            scenario="base",
            period_years=5,
            projected_annual_cash_flows=[100.0, 105.0],
            terminal_value=1000.0,
            present_value=800.0,
            iv_per_share=10.0,
            growth_rate=0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.01,
            margin_of_safety=0.50,
        )
        assert iv.iv_per_share == 10.0


class TestAnalysisResults:
    def test_construction(self) -> None:
        ar = AnalysisResults(
            time_series=pd.DataFrame(),
            companies=pd.DataFrame(),
            projections={},
            intrinsic_values={},
            growth_rankings=pd.DataFrame(),
            value_rankings=pd.DataFrame(),
            weighted_rankings=pd.DataFrame(),
            combined_rankings=pd.DataFrame(),
            factor_contributions=pd.DataFrame(),
            factor_dominance=pd.DataFrame(),
            quadrant_analysis=pd.DataFrame(),
            watchlist=["BHP"],
            filter_log=FilterLog(),
            live_prices={"BHP": 45.0},
            config=AnalysisConfig(),
        )
        assert ar.watchlist == ["BHP"]
