"""Tests for pipeline.config.settings."""

import pytest

from pipeline.config.settings import AnalysisConfig, ColumnSpec, TableSpec


class TestColumnSpec:
    def test_frozen(self) -> None:
        cs = ColumnSpec(db_column="revenue", internal_name="revenue")
        with pytest.raises(AttributeError):
            cs.db_column = "other"  # type: ignore[misc]

    def test_default_required(self) -> None:
        cs = ColumnSpec(db_column="a", internal_name="b")
        assert cs.required is True

    def test_optional(self) -> None:
        cs = ColumnSpec(db_column="a", internal_name="b", required=False)
        assert cs.required is False


class TestTableSpec:
    def test_valid_join_types(self) -> None:
        for jt in ("inner", "left"):
            ts = TableSpec(
                table_name="t",
                columns=(ColumnSpec(db_column="a", internal_name="b"),),
                join_type=jt,
            )
            assert ts.join_type == jt

    def test_invalid_join_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid join_type"):
            TableSpec(
                table_name="t",
                columns=(ColumnSpec(db_column="a", internal_name="b"),),
                join_type="outer",
            )


class TestAnalysisConfig:
    def test_defaults_valid(self) -> None:
        config = AnalysisConfig()
        assert config.market == "AU"
        assert config.mode == "shortlist"
        assert config.exchanges == ("ASX",)
        assert config.currencies == ("AUD",)

    def test_us_market(self) -> None:
        config = AnalysisConfig(market="US")
        assert config.exchanges == ("NASDAQ", "NYSE", "AMEX")
        assert config.currencies == ("USD",)

    def test_all_markets_valid(self) -> None:
        for market in ("US", "AU", "UK", "CA", "NZ", "SG", "HK"):
            config = AnalysisConfig(market=market)
            assert len(config.exchanges) >= 1
            assert len(config.currencies) >= 1

    def test_invalid_market(self) -> None:
        with pytest.raises(ValueError, match="Unknown market"):
            AnalysisConfig(market="XX")

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            AnalysisConfig(mode="invalid")

    def test_owned_mode_requires_companies(self) -> None:
        with pytest.raises(ValueError, match="non-empty owned_companies"):
            AnalysisConfig(mode="owned", owned_companies=[])

    def test_owned_mode_with_companies(self) -> None:
        config = AnalysisConfig(mode="owned", owned_companies=["BHP", "CBA"])
        assert config.owned_companies == ["BHP", "CBA"]

    def test_invalid_period_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid period_type"):
            AnalysisConfig(period_type="monthly")

    def test_discount_rate_must_exceed_terminal(self) -> None:
        with pytest.raises(ValueError, match="discount_rate"):
            AnalysisConfig(discount_rate=0.01, terminal_growth_rate=0.02)

    def test_margin_of_safety_bounds(self) -> None:
        with pytest.raises(ValueError, match="margin_of_safety"):
            AnalysisConfig(margin_of_safety=0.0)
        with pytest.raises(ValueError, match="margin_of_safety"):
            AnalysisConfig(margin_of_safety=1.0)

    def test_zero_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="base_fade_half_life_years"):
            AnalysisConfig(base_fade_half_life_years=0.0)

    def test_negative_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="base_fade_half_life_years"):
            AnalysisConfig(base_fade_half_life_years=-1.0)

    def test_growth_subweights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="Growth sub-weights"):
            AnalysisConfig(
                growth_rate_subweight=0.5,
                growth_stability_subweight=0.5,
                growth_divergence_subweight=0.5,
            )

    def test_fcf_revenue_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="fcf_growth_weight"):
            AnalysisConfig(fcf_growth_weight=0.5, revenue_growth_weight=0.3)

    def test_fiscal_year_range(self) -> None:
        config = AnalysisConfig(history_years=5)
        assert config.max_fiscal_year - config.min_fiscal_year == 5

    def test_default_table_specs(self) -> None:
        config = AnalysisConfig()
        assert len(config.table_specs) == 5
        table_names = [s.table_name for s in config.table_specs]
        assert "incomeStatement" in table_names
        assert "balanceSheet" in table_names
        assert "cashFlow" in table_names
        assert "cashFlowGrowth" in table_names
        assert "incomeStatementGrowth" in table_names
