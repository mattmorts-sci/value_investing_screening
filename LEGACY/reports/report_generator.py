"""
Comprehensive PDF report generator for intrinsic value analysis.

Generates detailed analysis reports with visualizations and insights.
Reports focus on FCF-based valuations with revenue tracked separately
for growth analysis and divergence assessment.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

from data_pipeline.data_structures import AnalysisResults, CompanyAnalysis
from config.user_config import AnalysisConfig
from reports.report_visualizations import (
    create_historical_growth_chart,
    create_growth_projection_chart,
    create_valuation_matrix,
    create_scenario_comparison,
    create_growth_value_scatter,
    create_financial_health_dashboard,
    create_ranking_comparison_table,
    create_growth_fan_chart,
    create_acquirers_multiple_analysis,
    create_projected_growth_stability_chart,
    create_risk_adjusted_opportunity_chart,
    create_comprehensive_rankings_table, 
    create_growth_comparison_historical,    
    create_growth_comparison_projected,
    create_valuation_upside_comparison      
)
from reports.methodology_generator import generate_methodology_document 

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive PDF analysis reports."""
    
    def __init__(self, results: AnalysisResults, config: AnalysisConfig):
        self.results = results
        self.config = config
        self.report_date = datetime.now()
    
    def generate_report(self, output_path: Path, detailed_companies: Optional[List[str]] = None):
        """
        Generate main analysis report PDF.
        
        Creates detailed analysis pages only for specified companies.
        All other companies get summary coverage only.
        
        Args:
            output_path: Path for output PDF
            detailed_companies: List of tickers for detailed reports 
                              (if None, uses top N from config)
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine which companies get detailed reports
        if detailed_companies is None:
            # Use top companies by opportunity rank
            detailed_companies = self.results.watchlist[:self.config.DETAILED_REPORT_COUNT]
            logger.info(f"Creating detailed reports for top {len(detailed_companies)} companies by opportunity rank")
        else:
            # Validate requested companies exist
            detailed_companies = [t for t in detailed_companies if t in self.results.companies]
            logger.info(f"Creating detailed reports for {len(detailed_companies)} specified companies")
        
        # Filter to companies with successful analysis
        valid_companies = [
            ticker for ticker in self.results.watchlist
            if ticker in self.results.companies and 
            self.results.companies[ticker].current_price is not None
        ]
        
        with PdfPages(str(output_path)) as pdf:
            # Title page
            self._add_title_page(pdf)
            
            # Executive summary
            self._add_executive_summary(pdf, valid_companies, detailed_companies)
            
            # Market overview
            self._add_market_overview(pdf)
            
            # Comparative analysis
            self._add_comparative_analysis(pdf, valid_companies)
            
            # Detailed individual company analyses (only for specified companies)
            for i, ticker in enumerate(detailed_companies, 1):
                if ticker in self.results.companies:
                    self._add_company_analysis(pdf, ticker, i)
                else:
                    logger.warning(f"Skipping detailed report for {ticker} - not found in results")
            
            # Summary table for all watchlist companies
            self._add_watchlist_summary(pdf, valid_companies)
            
            # Technical appendix
            self._add_technical_appendix(pdf)
            
            # Full rankings table
            self._add_full_rankings(pdf)
        
        logger.info(f"Report generated: {output_path}")
        
        # Generate methodology document
        methodology_path = output_path.parent / "methodology.pdf"
        generate_methodology_document(self.config, methodology_path)
    
    def _add_title_page(self, pdf: PdfPages):
        """Add title page to PDF."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.7, 'Intrinsic Value Analysis Report',
                ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Subtitle
        ax.text(0.5, 0.6, f'{self.config.COUNTRY} Market Analysis',
                ha='center', va='center', fontsize=18)
        
        # Date
        ax.text(0.5, 0.5, f'Analysis Date: {self.report_date.strftime("%B %d, %Y")}',
                ha='center', va='center', fontsize=14)
        
        # Data vintage
        ax.text(0.5, 0.45, f'Data Collection Date: {self.config.COLLECTION_DATE}',
                ha='center', va='center', fontsize=12)
        
        # Key parameters
        params_text = f"""
Key Analysis Parameters:
• Projection Period: {self.config.PRIMARY_PERIOD} years
• Discount Rate: {self.config.DISCOUNT_RATE:.1%}
• Margin of Safety: {self.config.MARGIN_OF_SAFETY:.0%}
• Terminal Growth: {self.config.TERMINAL_GROWTH_RATE:.1%}
• Divergence Threshold: {self.config.GROWTH_DIVERGENCE_THRESHOLD:.0%}

Valuation Methodology:
• FCF-based DCF only (revenue tracked separately for growth analysis)
        """
        ax.text(0.5, 0.3, params_text, ha='center', va='center', fontsize=11)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_executive_summary(self, pdf: PdfPages, valid_companies: List[str], 
                              detailed_companies: List[str]):
        """Add executive summary page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Executive Summary', 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # Analysis overview
        overview_text = f"""
Analysis Overview:
• Initial companies screened: {self.results.initial_company_count}
• Companies after filtering: {self.results.filtered_company_count}
• Companies with price data: {self.results.final_company_count}
• Final watchlist size: {len(valid_companies)}
• Detailed reports created for: {len(detailed_companies)} companies

Filtering Criteria Applied:
• Minimum Market Cap: ${self.config.MIN_MARKET_CAP:,.0f}
• Maximum Debt/Cash Ratio: {self.config.MAX_DEBT_TO_CASH_RATIO:.1f}

Valuation Approach:
• FCF-based DCF calculations only
• Revenue tracked for growth analysis but NOT used for valuation
• Growth divergence threshold: {self.config.GROWTH_DIVERGENCE_THRESHOLD:.0%}
        """
        ax.text(0.1, 0.85, overview_text, ha='left', va='top', fontsize=11)
        
        # Companies with detailed reports
        ax.text(0.1, 0.45, f'Companies with Detailed Analysis ({len(detailed_companies)}):', 
                ha='left', va='top', fontsize=14, fontweight='bold')
        
        y_pos = 0.40
        for i, ticker in enumerate(detailed_companies, 1):
            if ticker in self.results.companies:
                company = self.results.companies[ticker]
                summary = self._get_company_summary(ticker, company)
                ax.text(0.1, y_pos, f"{i}. {summary}", 
                        ha='left', va='top', fontsize=10)
                y_pos -= 0.06
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_company_analysis(self, pdf: PdfPages, ticker: str, rank: int):
        """Add detailed analysis for a single company."""
        company = self.results.companies[ticker]
        
        # Page 1: Overview and growth analysis
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f'#{rank} - {ticker} Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Create grid
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], hspace=0.3, wspace=0.3)
        
        # Company metrics summary
        ax_summary = fig.add_subplot(gs[0, :])
        self._add_company_metrics_table(ax_summary, ticker, company)
        
        # Growth fan chart
        ax_fan = fig.add_subplot(gs[1, 0])
        create_growth_fan_chart(company, ax_fan, self.config)
        
        # FCF valuation scenarios (using viridis)
        ax_val = fig.add_subplot(gs[1, 1])
        create_valuation_matrix(company, ax_val, self.config)
        
        # Historical growth patterns
        ax_hist = fig.add_subplot(gs[2, 0])
        create_historical_growth_chart(company, ax_hist, self.config)
        
        # Financial health
        ax_health = fig.add_subplot(gs[2, 1])
        create_financial_health_dashboard(company, ax_health, self.config)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Scenario analysis
        fig2 = create_scenario_comparison(company, self.config)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
    
    def _add_watchlist_summary(self, pdf: PdfPages, valid_companies: List[str]):
        """Add summary table for all watchlist companies."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ax.text(0.5, 0.98, 'Complete Watchlist Summary', 
                ha='center', va='top', fontsize=16, fontweight='bold', 
                transform=ax.transAxes)
        
        # Prepare data
        rows = []
        for ticker in valid_companies[:30]:  # Limit to fit on page
            if ticker not in self.results.companies:
                continue
            
            company = self.results.companies[ticker]
            period = self.config.PRIMARY_PERIOD
            
            # Get key metrics
            fcf_proj = company.get_projection(period, 'fcf', 'median')
            rev_proj = company.get_projection(period, 'revenue', 'median')
            divergence = company.growth_divergence or 0.0
            
            # Get opportunity rank
            opp_rank = self.results.combined_rankings[
                self.results.combined_rankings['ticker'] == ticker
            ]['opportunity_rank'].values[0]
            
            # Format divergence with warning
            div_text = f'{divergence:.1%}'
            if divergence > self.config.GROWTH_DIVERGENCE_THRESHOLD:
                div_text = f'⚠️ {div_text}'
            
            rows.append([
                ticker.split(':')[0],
                f'${company.current_price:.2f}',
                f'{fcf_proj.annual_growth_median:.1%}',
                f'{rev_proj.annual_growth_median:.1%}',
                div_text,
                f'{company.best_iv_to_price_ratio:.1f}x',
                f'#{int(opp_rank)}'
            ])
        
        # Create table
        columns = ['Ticker', 'Price', 'FCF Gr.', 'Rev Gr.', 'Diverg.', 'IV/Price', 'Rank']
        
        table = ax.table(cellText=rows,
                        colLabels=columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.05, 1, 0.90])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight divergence warnings
        for i in range(1, len(rows) + 1):
            if '⚠️' in rows[i-1][4]:
                table[(i, 4)].set_facecolor('#FFEB3B')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_technical_appendix(self, pdf: PdfPages):
        """Add technical appendix with methodology notes."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Technical Appendix', 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # DCF methodology explanation
        dcf_text = f"""
DCF Valuation Methodology:

Why FCF-Only Valuation:
• DCF (Discounted Cash Flow) models should only use actual cash flows
• Revenue is an accounting measure, not a cash flow
• Free Cash Flow (FCF) represents actual cash available to investors
• Revenue growth is tracked separately as a business quality indicator

Growth Divergence Analysis:
• FCF and revenue growth rates are compared using a flat threshold
• Divergence > {self.config.GROWTH_DIVERGENCE_THRESHOLD:.0%} triggers a warning flag
• We use a flat threshold (not relative) because a 10% divergence between FCF 
  and revenue growth indicates business issues regardless of absolute growth rates
• High divergence may indicate:
  - Margin compression (revenue growing faster than FCF)
  - Unsustainable cost cutting (FCF growing faster than revenue)
  - Business model changes
• For example: FCF growing at 2% with revenue at 12% is as concerning as 
  FCF at 25% with revenue at 15% - both represent 10% divergence

Model Assumptions:
• Projection Period: {self.config.PRIMARY_PERIOD} years
• Discount Rate (WACC): {self.config.DISCOUNT_RATE:.1%}
• Terminal Growth Rate: {self.config.TERMINAL_GROWTH_RATE:.1%}
• Margin of Safety: {self.config.MARGIN_OF_SAFETY:.0%}
• Quarterly to Annual Conversion: {self.config.QUARTERS_PER_YEAR} quarters/year

Monte Carlo Simulation:
• Simulations per metric: {self.config.SIMULATION_REPLICATES:,}
• Growth constraints: [{self.config.MIN_QUARTERLY_GROWTH:.1%}, {self.config.MAX_QUARTERLY_GROWTH:.1%}] quarterly
• Scenarios: Conservative (25th %ile), Base (50th %ile), Optimistic (75th %ile)
        """
        
        ax.text(0.1, 0.90, dcf_text, ha='left', va='top', fontsize=10, 
                family='monospace', wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_market_overview(self, pdf: PdfPages):
        """Add market overview visualizations."""
        # First page: standard 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Market Overview', fontsize=16, fontweight='bold')
        
        # 1. Size distribution
        self._plot_size_distribution(axes[0, 0])
        
        # 2. Growth distribution
        self._plot_growth_distribution(axes[0, 1])
        
        # 3. Valuation distribution
        self._plot_valuation_distribution(axes[1, 0])
        
        # 4. Filter impact
        self._plot_filter_impact(axes[1, 1])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Second page: Historical Growth vs Stability Analysis
        # Get top companies by risk-adjusted rank
        risk_adj_sorted = self.results.combined_rankings.sort_values('risk_adjusted_rank')
        top_companies = risk_adj_sorted.head(30)['ticker'].tolist()
        
        fig2 = create_projected_growth_stability_chart(self.results, top_companies, self.config)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # Third page: Acquirer's Multiple Analysis
        fig3 = create_acquirers_multiple_analysis(self.results, self.results.watchlist, self.config)
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)
    
    def _add_comparative_analysis(self, pdf: PdfPages, valid_companies: List[str]):
        """Add comparative analysis section."""
        # Get top companies by risk-adjusted rank
        risk_adj_sorted = self.results.combined_rankings.sort_values('risk_adjusted_rank')
        top_risk_adjusted = risk_adj_sorted.head(30)['ticker'].tolist()
        
        # Valuation Upside/Downside Comparison
        fig_valuation = create_valuation_upside_comparison(self.results, valid_companies, self.config)
        pdf.savefig(fig_valuation, bbox_inches='tight')
        plt.close(fig_valuation)

        # Growth-Value Matrix with proper formatting
        fig1 = create_growth_value_scatter(self.results, valid_companies, self.config)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Risk-Adjusted Opportunity Analysis
        fig2 = create_risk_adjusted_opportunity_chart(self.results, top_risk_adjusted, self.config)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # Growth Comparison - Both Historical and Projected
        # First show historical (actual past performance)
        fig3_hist = create_growth_comparison_historical(results)
        pdf.savefig(fig3_hist, bbox_inches='tight')
        plt.close(fig3_hist)
        
        # Then show projected (future expectations with uncertainty)
        fig3_proj = create_growth_comparison_projected(results)
        pdf.savefig(fig3_proj, bbox_inches='tight')
        plt.close(fig3_proj)
        
        # Comprehensive Rankings Table
        fig4 = create_comprehensive_rankings_table(self.results, top_risk_adjusted, self.config)
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)
        
        # Growth Projection Chart for top companies
        fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig5.suptitle('Growth Projections - Top 4 Risk-Adjusted Companies', fontsize=16, fontweight='bold')
        
        for i, (ax, ticker) in enumerate(zip(axes.flat, top_risk_adjusted[:4])):
            if ticker in self.results.companies:
                create_growth_projection_chart(self.results.companies[ticker], ax, self.config)
                ax.set_title(f'{ticker}', fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig5, bbox_inches='tight')
        plt.close(fig5)
        
        # Ranking comparison table - now shows risk-adjusted rankings
        fig6 = create_ranking_comparison_table(self.results, top_risk_adjusted[:15], self.config)
        pdf.savefig(fig6, bbox_inches='tight')
        plt.close(fig6)
        
        # Acquirer's Multiple Analysis
        fig7 = create_acquirers_multiple_analysis(self.results, valid_companies, self.config)
        pdf.savefig(fig7, bbox_inches='tight')
        plt.close(fig7)
    
    def _add_full_rankings(self, pdf: PdfPages):
        """Add full rankings table."""
        # Prepare data
        df = self.results.combined_rankings.copy()
        df = df[df['ticker'].isin(self.results.watchlist[:40])]
        
        # Select key columns including divergence
        columns = ['ticker', 'current_price', 'fcf_growth_annual', 
                  'revenue_growth_annual', 'growth_divergence', 'best_iv_ratio', 
                  'growth_stability', 'opportunity_rank']
        
        # Format for display
        display_df = df[columns].copy()
        display_df['fcf_growth_annual'] = (display_df['fcf_growth_annual'] * 100).round(1).astype(str) + '%'
        display_df['revenue_growth_annual'] = (display_df['revenue_growth_annual'] * 100).round(1).astype(str) + '%'
        display_df['growth_divergence'] = (display_df['growth_divergence'] * 100).round(1).astype(str) + '%'
        display_df['best_iv_ratio'] = display_df['best_iv_ratio'].round(2)
        display_df['growth_stability'] = display_df['growth_stability'].round(2)
        display_df['opportunity_rank'] = display_df['opportunity_rank'].astype(int)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        ax.text(0.5, 0.98, 'Complete Rankings Table (FCF-Based Valuation)', 
                ha='center', va='top', fontsize=16, fontweight='bold', 
                transform=ax.transAxes)
        
        # Create table
        table = ax.table(cellText=display_df.values,
                        colLabels=['Ticker', 'Price', 'FCF Growth', 
                                  'Rev Growth', 'Divergence', 'IV/Price', 'Stability', 'Rank'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 0.95])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # Helper methods
    def _get_company_summary(self, ticker: str, company: CompanyAnalysis) -> str:
        """Get one-line summary for a company."""
        period = self.config.PRIMARY_PERIOD
        
        # Get FCF growth
        fcf_proj = company.get_projection(period, 'fcf', 'median')
        fcf_growth = fcf_proj.annual_growth_median
        
        # Get divergence
        divergence = company.growth_divergence or 0.0
        div_text = f", Divergence={divergence:.1%}" if divergence > self.config.GROWTH_DIVERGENCE_THRESHOLD else ""
        
        return (f"{ticker}: IV/Price={company.best_iv_to_price_ratio:.2f}x, "
                f"FCF Growth={fcf_growth:.1%}, "
                f"Market Cap=${company.company_data.market_cap/1e6:.0f}M{div_text}")
    
    def _add_company_metrics_table(self, ax, ticker: str, company: CompanyAnalysis):
        """Add company metrics summary table."""
        ax.axis('off')
        
        # Get growth projections
        period = self.config.PRIMARY_PERIOD
        fcf_proj = company.get_projection(period, 'fcf', 'median')
        rev_proj = company.get_projection(period, 'revenue', 'median')
        fcf_growth = fcf_proj.annual_growth_median
        rev_growth = rev_proj.annual_growth_median
        divergence = company.growth_divergence or 0.0
        
        # Prepare metrics
        metrics = [
            ['Current Price', f'${company.current_price:.2f}'],
            ['Market Cap', f'${company.company_data.market_cap/1e6:.0f}M'],
            ['FCF', f'${company.company_data.fcf/1e6:.0f}M'],
            ['FCF Growth', f'{fcf_growth:.1%}'],
            ['Revenue Growth', f'{rev_growth:.1%}'],
            ['Growth Divergence', f'{divergence:.1%}' + (' ⚠️' if divergence > self.config.GROWTH_DIVERGENCE_THRESHOLD else '')],
            ['FCF IV/Price', f'{company.best_iv_to_price_ratio:.2f}x'],
            ['Debt/Cash', f'{company.company_data.lt_debt/company.company_data.cash_and_equiv:.2f}'],
            ['Growth Stability', f'{company.growth_stability_score:.2f}']
        ]
        
        # Create table
        table = ax.table(cellText=metrics,
                        colWidths=[0.5, 0.5],
                        cellLoc='left',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
    
    def _plot_size_distribution(self, ax):
        """Plot market cap distribution."""
        market_caps = [c.company_data.market_cap/1e9 
                      for c in self.results.companies.values()]
        
        ax.hist(market_caps, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Market Cap ($B)')
        ax.set_ylabel('Count')
        ax.set_title('Market Cap Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_growth_distribution(self, ax):
        """Plot growth rate distribution."""
        growth_rates = []
        period = self.config.PRIMARY_PERIOD
        
        for c in self.results.companies.values():
            fcf_proj = c.get_projection(period, 'fcf', 'median')
            growth_rates.append(fcf_proj.annual_growth_median * 100)
        
        ax.hist(growth_rates, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('FCF Growth Rate (%)')
        ax.set_ylabel('Count')
        ax.set_title('Growth Rate Distribution')
        ax.axvline(x=10, color='red', linestyle='--', label='10% Target')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_valuation_distribution(self, ax):
        """Plot IV/Price distribution."""
        ratios = [c.best_iv_to_price_ratio 
                 for c in self.results.companies.values()
                 if c.best_iv_to_price_ratio is not None]
        
        # Cap at 5 for visualization
        ratios = [min(r, 5) for r in ratios]
        
        ax.hist(ratios, bins=20, edgecolor='black', alpha=0.7, color='blue')
        ax.set_xlabel('FCF IV/Price Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Valuation Distribution')
        ax.axvline(x=1.0, color='red', linestyle='--', label='Fair Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_filter_impact(self, ax):
        """Plot filtering funnel."""
        stages = ['Initial', 'Financial\nFilters', 'Growth\nFilters', 'With\nPrices', 'Watchlist']
        counts = [
            self.results.initial_company_count,
            self.results.filtered_company_count,
            self.results.filtered_company_count,  # Same as financial for now
            self.results.final_company_count,
            len(self.results.watchlist)
        ]
        
        ax.bar(stages, counts, color=['gray', 'orange', 'yellow', 'lightgreen', 'darkgreen'])
        ax.set_ylabel('Number of Companies')
        ax.set_title('Filtering Funnel')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        ax.grid(True, axis='y', alpha=0.3)


def generate_comprehensive_report(results: AnalysisResults, config: AnalysisConfig, 
                                output_dir: Path, detailed_companies: Optional[List[str]] = None):
    """
    Generate comprehensive PDF analysis report.
    
    Creates detailed analysis only for specified companies to keep
    report size manageable while still providing overview of all opportunities.
    
    Args:
        results: Analysis results
        config: Analysis configuration
        output_dir: Output directory
        detailed_companies: List of tickers for detailed analysis
                          (if None, uses top N from config)
    """
    generator = ReportGenerator(results, config)
    
    report_path = output_dir / f"analysis_report_{config.COUNTRY}_{config.COLLECTION_DATE}.pdf"
    generator.generate_report(report_path, detailed_companies)
    
    logger.info(f"Report generation complete: {report_path}")
    
    return report_path


# Integration with main pipeline
if __name__ == "__main__":
    # Example usage
    from _refactored.scripts.main_pipeline import run_analysis
    from config.user_config import AnalysisConfig
    
    config = AnalysisConfig()
    results = run_analysis(config)
    
    output_dir = config.output_directory
    
    # Example: Create detailed reports only for specific companies
    detailed_companies = ['BHP:AU', 'CBA:AU', 'CSL:AU']  # Or None for top 5
    
    generate_comprehensive_report(results, config, output_dir, detailed_companies)