"""Pipeline data contracts.

Dataclasses defining the shape of data passed between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config.settings import AnalysisConfig


@dataclass
class RawFinancialData:
    """Output of the data loader."""

    data: pd.DataFrame
    query_metadata: dict[str, Any]
    row_count: int
    company_count: int
    period_range: tuple[int, int]
    dropped_companies_path: Path


@dataclass
class FilterLog:
    """Track which companies are removed by each filter."""

    removed: dict[str, list[str]] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    owned_bypassed: list[str] = field(default_factory=list)
    owned_tracking: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Projection:
    """Growth projection for one company, one metric, one scenario."""

    entity_id: int
    metric: str
    period_years: int
    scenario: str
    quarterly_growth_rates: list[float]
    quarterly_values: list[float]
    annual_cagr: float
    current_value: float


@dataclass
class IntrinsicValue:
    """DCF valuation result for one scenario."""

    scenario: str
    period_years: int
    projected_annual_cash_flows: list[float]
    terminal_value: float
    present_value: float
    iv_per_share: float
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    margin_of_safety: float


@dataclass
class AnalysisResults:
    """Complete pipeline output."""

    time_series: pd.DataFrame
    companies: pd.DataFrame
    projections: dict[int, dict[str, Any]]
    intrinsic_values: dict[int, dict[str, Any]]
    growth_rankings: pd.DataFrame
    value_rankings: pd.DataFrame
    weighted_rankings: pd.DataFrame
    combined_rankings: pd.DataFrame
    factor_contributions: pd.DataFrame
    factor_dominance: pd.DataFrame
    quadrant_analysis: pd.DataFrame
    watchlist: list[str]
    filter_log: FilterLog
    live_prices: dict[str, float]
    config: AnalysisConfig
