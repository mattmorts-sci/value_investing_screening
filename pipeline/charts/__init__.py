"""Chart functions for value investing analysis.

All chart functions take AnalysisResults and return matplotlib Figure objects.
Organised into three modules by scope:

- market_overview: Market-wide charts (6 functions)
- comparative: Multi-company comparison charts (5 functions)
- company_detail: Per-company detail charts (7 functions)
- tables: Summary table rendering (2 functions)
"""

from pipeline.charts.company_detail import (
    financial_health_dashboard,
    growth_fan,
    growth_projection,
    historical_growth,
    risk_spider,
    scenario_comparison,
    valuation_matrix,
)
from pipeline.charts.comparative import (
    acquirers_multiple_analysis,
    comprehensive_rankings_table,
    projected_growth_stability,
    ranking_comparison_table,
    valuation_upside_comparison,
)
from pipeline.charts.market_overview import (
    factor_contributions_bar,
    factor_heatmap,
    growth_comparison_historical,
    growth_comparison_projected,
    growth_value_scatter,
    risk_adjusted_opportunity,
)
from pipeline.charts.tables import (
    filter_summary,
    watchlist_summary,
)

__all__ = [
    "acquirers_multiple_analysis",
    "comprehensive_rankings_table",
    "factor_contributions_bar",
    "factor_heatmap",
    "filter_summary",
    "financial_health_dashboard",
    "growth_comparison_historical",
    "growth_comparison_projected",
    "growth_fan",
    "growth_projection",
    "growth_value_scatter",
    "historical_growth",
    "projected_growth_stability",
    "ranking_comparison_table",
    "risk_adjusted_opportunity",
    "risk_spider",
    "scenario_comparison",
    "valuation_matrix",
    "valuation_upside_comparison",
    "watchlist_summary",
]
