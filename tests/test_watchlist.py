"""Tests for pipeline.analysis.watchlist."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.analysis.watchlist import select_watchlist
from pipeline.config.settings import AnalysisConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rankings(n: int = 10) -> pd.DataFrame:
    """Create a rankings DataFrame with n companies.

    Sorted by opportunity_rank ascending. IV ratios descend with rank.
    """
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "entity_id": i,
            "symbol": f"CO{i}",
            "composite_iv_ratio": 5.0 - i * 0.4,
            "opportunity_rank": i,
        })
    return pd.DataFrame(rows).set_index("entity_id")


# ---------------------------------------------------------------------------
# Two-step selection
# ---------------------------------------------------------------------------

class TestTwoStepSelection:

    def test_basic_selection(self) -> None:
        rankings = _make_rankings(10)
        config = AnalysisConfig(
            iv_prefilter_count=100, target_watchlist_size=5,
        )
        result = select_watchlist(rankings, config)

        assert len(result) == 5
        # Top 5 by opportunity_rank (since all pass IV pre-filter)
        assert result == ["CO1", "CO2", "CO3", "CO4", "CO5"]

    def test_iv_prefilter_limits_candidates(self) -> None:
        """Only top N by IV ratio enter Step 2."""
        rankings = _make_rankings(10)
        config = AnalysisConfig(
            iv_prefilter_count=3, target_watchlist_size=5,
        )
        result = select_watchlist(rankings, config)

        # Only 3 pass IV pre-filter, so watchlist <= 3
        assert len(result) <= 3
        # All should be from top-3 IV ratio companies
        top3_symbols = set(
            rankings.nlargest(3, "composite_iv_ratio")["symbol"],
        )
        assert set(result).issubset(top3_symbols)

    def test_sorted_by_opportunity_rank(self) -> None:
        rankings = _make_rankings(10)
        config = AnalysisConfig(
            iv_prefilter_count=100, target_watchlist_size=5,
        )
        result = select_watchlist(rankings, config)

        # Verify ordering: CO1 (rank 1) before CO5 (rank 5)
        for i in range(len(result) - 1):
            idx_a = rankings[rankings["symbol"] == result[i]].index[0]
            idx_b = rankings[rankings["symbol"] == result[i + 1]].index[0]
            rank_a = rankings.loc[idx_a, "opportunity_rank"]
            rank_b = rankings.loc[idx_b, "opportunity_rank"]
            assert rank_a <= rank_b

    def test_fewer_companies_than_target(self) -> None:
        """Watchlist can be smaller than target if not enough companies."""
        rankings = _make_rankings(3)
        config = AnalysisConfig(
            iv_prefilter_count=100, target_watchlist_size=10,
        )
        result = select_watchlist(rankings, config)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# Owned-company inclusion
# ---------------------------------------------------------------------------

class TestOwnedCompanyInclusion:

    def test_owned_always_included(self) -> None:
        rankings = _make_rankings(10)
        # CO10 has worst IV ratio and worst rank
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["CO10"],
            iv_prefilter_count=3,
            target_watchlist_size=2,
        )
        result = select_watchlist(rankings, config)

        assert "CO10" in result

    def test_owned_already_selected_not_duplicated(self) -> None:
        rankings = _make_rankings(10)
        # CO1 would already be selected (best rank + best IV)
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["CO1"],
            iv_prefilter_count=100,
            target_watchlist_size=5,
        )
        result = select_watchlist(rankings, config)

        assert result.count("CO1") == 1

    def test_shortlist_mode_does_not_force_include(self) -> None:
        rankings = _make_rankings(10)
        config = AnalysisConfig(
            mode="shortlist",
            iv_prefilter_count=3,
            target_watchlist_size=2,
        )
        result = select_watchlist(rankings, config)

        # CO10 not included since shortlist mode doesn't force inclusion
        assert "CO10" not in result

    def test_owned_not_in_rankings(self) -> None:
        """Owned company not in rankings at all — not added."""
        rankings = _make_rankings(5)
        config = AnalysisConfig(
            mode="owned",
            owned_companies=["MISSING"],
            iv_prefilter_count=100,
            target_watchlist_size=3,
        )
        result = select_watchlist(rankings, config)

        assert "MISSING" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestWatchlistEdgeCases:

    def test_empty_rankings(self) -> None:
        rankings = pd.DataFrame(
            columns=["symbol", "composite_iv_ratio", "opportunity_rank"],
        )
        rankings.index.name = "entity_id"
        config = AnalysisConfig()
        result = select_watchlist(rankings, config)

        assert result == []

    def test_missing_columns_raises(self) -> None:
        rankings = pd.DataFrame(
            {"symbol": ["A"]}, index=pd.Index([1], name="entity_id"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            select_watchlist(rankings, AnalysisConfig())
