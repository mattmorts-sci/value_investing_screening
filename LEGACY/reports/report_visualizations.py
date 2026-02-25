"""
Visualization functions for the analysis report.

Creates clear, informative charts focused on value investing insights.
All functions require config parameter to access analysis settings.
All charts use viridis colormap for consistency.
Growth rates are displayed as CAGR (annual rates) calculated from quarterly data.
"""
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

from data_pipeline.data_structures import CompanyAnalysis, AnalysisResults
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def create_historical_growth_chart(company: CompanyAnalysis, ax: Optional[plt.Axes] = None, config: Optional[AnalysisConfig] = None):
    """
    Create historical FCF and revenue growth visualization.
    
    Shows historical CAGR calculated from point-to-point over the historical period.
    
    Args:
        company: Company analysis object
        ax: Optional matplotlib axes
        config: Analysis configuration
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get historical CAGR values (already calculated in data loading)
    fcf_cagr = company.company_data.historical_fcf_cagr * 100  # Convert to percentage
    rev_cagr = company.company_data.historical_revenue_cagr * 100
    
    # Handle extreme values for display
    fcf_display = f"{fcf_cagr:.1f}%" if abs(fcf_cagr) <= 100 else (">100%" if fcf_cagr > 0 else "<-100%")
    rev_display = f"{rev_cagr:.1f}%" if abs(rev_cagr) <= 100 else (">100%" if rev_cagr > 0 else "<-100%")
    
    # Create bar chart for CAGR values
    metrics = ['FCF', 'Revenue']
    values = [fcf_cagr, rev_cagr]
    
    # Cap display values at ±100% for chart scale
    display_values = [max(-100, min(100, v)) for v in values]
    
    colors = plt.cm.viridis([0.2, 0.8])
    bars = ax.bar(metrics, display_values, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, actual_val, display_text in zip(bars, values, [fcf_display, rev_display]):
        height = bar.get_height()
        y_pos = height + 2 if height >= 0 else height - 2
        ax.text(bar.get_x() + bar.get_width()/2., y_pos, display_text,
                ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Add growth variance information
    fcf_vol = np.sqrt(company.company_data.fcf_growth_variance) * 2 * 100  # Annualized
    rev_vol = np.sqrt(company.company_data.revenue_growth_variance) * 2 * 100
    
    # Add text box with volatility info
    textstr = f'Annual Volatility:\nFCF: {fcf_vol:.1f}%\nRevenue: {rev_vol:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.set_ylabel('Historical CAGR (%)', fontsize=12)
    ax.set_title(f'Historical Growth Rates (Point-to-Point CAGR)\n{company.company_data.ticker}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-100, 100)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.3)
    
    return ax


def create_growth_projection_chart(company: CompanyAnalysis, ax: Optional[plt.Axes] = None, config: AnalysisConfig = None):
    """
    Create growth projection visualization with confidence intervals.
    
    Shows projected growth rates from Monte Carlo simulation.
    Clearly distinguishes between historical CAGR and projected growth.
    
    Args:
        company: Company analysis object
        ax: Optional matplotlib axes
        config: Analysis configuration (required)
    """
    if config is None:
        logger.error("Config parameter is required for growth projection chart")
        raise ValueError("Config parameter is required")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Time series
    years = np.arange(0, config.PRIMARY_PERIOD + 1)
    
    # Get projections for the primary period
    period = config.PRIMARY_PERIOD
    logger.debug(f"Getting projections for {company.company_data.ticker} for {period} years")
    
    fcf_proj = company.get_projection(period, 'fcf', 'median')
    fcf_proj_top = company.get_projection(period, 'fcf', 'top')
    fcf_proj_bottom = company.get_projection(period, 'fcf', 'bottom')
    rev_proj = company.get_projection(period, 'revenue', 'median')
    
    # FCF projections using projected growth rates
    fcf_current = company.company_data.fcf
    fcf_median = fcf_current * (1 + fcf_proj.annual_growth_median) ** years
    fcf_top = fcf_current * (1 + fcf_proj_top.annual_growth_ci_top) ** years
    fcf_bottom = fcf_current * (1 + fcf_proj_bottom.annual_growth_ci_bottom) ** years
    
    # Plot with viridis colors
    colors = plt.cm.viridis([0.2, 0.8])
    ax.plot(years, fcf_median / 1e6, color=colors[0], linewidth=2, 
            label=f'FCF Projected (CAGR: {fcf_proj.annual_growth_median:.1%})')
    ax.fill_between(years, fcf_bottom / 1e6, fcf_top / 1e6, 
                    alpha=0.3, color=colors[0], label='FCF 25th-75th %ile')
    
    # Revenue projections
    rev_current = company.company_data.revenue
    rev_median = rev_current * (1 + rev_proj.annual_growth_median) ** years
    
    ax.plot(years, rev_median / 1e6, color=colors[1], linestyle='--', linewidth=2, 
            label=f'Revenue Projected (CAGR: {rev_proj.annual_growth_median:.1%})')
    
    # Add historical CAGR annotation for comparison
    hist_fcf_cagr = company.company_data.historical_fcf_cagr * 100
    hist_rev_cagr = company.company_data.historical_revenue_cagr * 100
    
    # Handle extreme values for display
    fcf_hist_text = f"{hist_fcf_cagr:.1f}%" if abs(hist_fcf_cagr) <= 100 else (">100%" if hist_fcf_cagr > 0 else "<-100%")
    rev_hist_text = f"{hist_rev_cagr:.1f}%" if abs(hist_rev_cagr) <= 100 else (">100%" if hist_rev_cagr > 0 else "<-100%")
    
    textstr = f'Historical CAGR:\nFCF: {fcf_hist_text}\nRevenue: {rev_hist_text}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Years')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Future Growth Projections (Monte Carlo Simulation)\nvs Historical CAGR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_valuation_matrix(company: CompanyAnalysis, ax: Optional[plt.Axes] = None, config: AnalysisConfig = None):
    """
    Create FCF valuation scenarios visualization.
    
    Shows FCF-based intrinsic values across three scenarios.
    Note: Revenue is tracked separately but not used for valuation.
    
    Args:
        company: Company analysis object
        ax: Optional matplotlib axes
        config: Analysis configuration (required)
    """
    if config is None:
        logger.error("Config parameter is required for valuation matrix")
        raise ValueError("Config parameter is required")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract FCF intrinsic values for primary period
    period = config.PRIMARY_PERIOD
    scenarios = ['bottom', 'median', 'top']
    scenario_labels = ['Conservative\n(25th %ile)', 'Base Case\n(50th %ile)', 'Optimistic\n(75th %ile)']
    
    values = []
    for scenario in scenarios:
        logger.debug(f"Getting FCF intrinsic value for {scenario} at {period} years")
        iv = company.get_intrinsic_value(period, 'fcf', scenario)
        values.append(iv)
    
    # Create bar chart with viridis colors
    x = np.arange(len(scenarios))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scenarios)))
    bars = ax.bar(x, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add current price line if available
    if company.current_price:
        ax.axhline(y=company.current_price, color='red', linestyle='--', linewidth=2,
                  label=f'Current Price: ${company.current_price:.2f}')
        
        # Show upside/downside for base case
        base_iv = values[1]  # median scenario
        upside = (base_iv / company.current_price - 1) * 100
        upside_text = f"Base Case Upside: {upside:.0f}%" if upside > 0 else f"Base Case Downside: {upside:.0f}%"
        ax.text(0.5, 0.95, upside_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if upside > 0 else 'lightcoral', alpha=0.7))
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel('FCF-Based Intrinsic Value per Share ($)', fontsize=12)
    ax.set_title(f'FCF Valuation Scenarios ({period} Year DCF)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    if company.current_price:
        ax.legend(loc='upper right')
    
    return ax


def create_growth_fan_chart(company: CompanyAnalysis, ax: Optional[plt.Axes] = None, config: AnalysisConfig = None):
    """
    Create growth trajectory fan chart showing FCF and revenue projections.
    
    This visualization displays the range of possible growth outcomes as a fan,
    with the median projection as a solid line and confidence intervals as shaded areas.
    It helps visualize uncertainty in growth projections and compare FCF vs revenue trajectories.
    
    Growth rates are shown as CAGR calculated from quarterly projections.
    
    Args:
        company: Company analysis object
        ax: Optional matplotlib axes (if None, creates new figure)
        config: Analysis configuration (required)
    """
    if config is None:
        logger.error("Config parameter is required for growth fan chart")
        raise ValueError("Config parameter is required")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Time series for primary period
    period = config.PRIMARY_PERIOD
    years = np.linspace(0, period, period + 1)
    
    logger.debug(f"Creating growth fan chart for {company.company_data.ticker}")
    
    # Get FCF projections for all scenarios
    fcf_bottom = company.get_projection(period, 'fcf', 'bottom')
    fcf_median = company.get_projection(period, 'fcf', 'median')
    fcf_top = company.get_projection(period, 'fcf', 'top')
    
    # Get revenue projections for all scenarios
    rev_bottom = company.get_projection(period, 'revenue', 'bottom')
    rev_median = company.get_projection(period, 'revenue', 'median')
    rev_top = company.get_projection(period, 'revenue', 'top')
    
    # Calculate trajectories from current values using CAGR
    fcf_current = company.company_data.fcf / 1e6  # Convert to millions
    rev_current = company.company_data.revenue / 1e6
    
    # FCF trajectories (annual rates are already CAGR)
    fcf_traj_bottom = fcf_current * (1 + fcf_bottom.annual_growth_ci_bottom) ** years
    fcf_traj_median = fcf_current * (1 + fcf_median.annual_growth_median) ** years
    fcf_traj_top = fcf_current * (1 + fcf_top.annual_growth_ci_top) ** years
    
    # Revenue trajectories (annual rates are already CAGR)
    rev_traj_bottom = rev_current * (1 + rev_bottom.annual_growth_ci_bottom) ** years
    rev_traj_median = rev_current * (1 + rev_median.annual_growth_median) ** years
    rev_traj_top = rev_current * (1 + rev_top.annual_growth_ci_top) ** years
    
    # Plot with viridis colors
    colors = plt.cm.viridis([0.2, 0.8])
    
    # Plot FCF fan
    ax.fill_between(years, fcf_traj_bottom, fcf_traj_top, 
                    alpha=0.3, color=colors[0], label='FCF Range (25th-75th %ile)')
    ax.plot(years, fcf_traj_median, color=colors[0], linewidth=2, 
            label=f'FCF Median (CAGR: {fcf_median.annual_growth_median:.1%})')
    
    # Plot revenue fan
    ax.fill_between(years, rev_traj_bottom, rev_traj_top, 
                    alpha=0.3, color=colors[1], label='Revenue Range (25th-75th %ile)')
    ax.plot(years, rev_traj_median, color=colors[1], linestyle='--', linewidth=2, 
            label=f'Revenue Median (CAGR: {rev_median.annual_growth_median:.1%})')
    
    # Add starting points
    ax.scatter([0], [fcf_current], color=colors[0], s=100, zorder=5, marker='o')
    ax.scatter([0], [rev_current], color=colors[1], s=100, zorder=5, marker='s')
    
    # Calculate and display divergence
    divergence = abs(fcf_median.annual_growth_median - rev_median.annual_growth_median)
    if divergence > config.GROWTH_DIVERGENCE_THRESHOLD:
        divergence_text = f"* Growth Divergence: {divergence:.1%}"
        ax.text(0.95, 0.05, divergence_text, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Value ($M)', fontsize=12)
    ax.set_title(f'Growth Trajectory Fan Chart - {company.company_data.ticker}\n(CAGR from Quarterly Projections)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start at 0 for better comparison
    ax.set_ylim(bottom=0)
    
    return ax


def create_scenario_comparison(company: CompanyAnalysis, config: AnalysisConfig):
    """
    Create detailed scenario comparison visualization.
    
    Args:
        company: Company analysis object
        config: Analysis configuration (required)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{company.company_data.ticker} - Scenario Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Waterfall chart - IV components
    ax1 = axes[0, 0]
    _create_dcf_waterfall(ax1, company, config)
    
    # 2. Sensitivity analysis
    ax2 = axes[0, 1]
    _create_sensitivity_analysis(ax2, company, config)
    
    # 3. Growth trajectory comparison
    ax3 = axes[1, 0]
    _create_growth_trajectories(ax3, company, config)
    
    # 4. Risk-return profile
    ax4 = axes[1, 1]
    _create_risk_return_profile(ax4, company, config)
    
    plt.tight_layout()
    return fig


def create_growth_value_scatter(results: AnalysisResults, companies: List[str], config: AnalysisConfig):
    """
    Create growth vs value opportunity scatter plot.
    
    Shows only specified companies (no grey background points).
    Growth values shown are PROJECTED CAGR from Monte Carlo simulations.
    Adapts quadrant labels and interpretation based on ANALYSIS_OBJECTIVE.
    
    Args:
        results: Analysis results
        companies: List of company tickers to display
        config: Analysis configuration (required)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Determine analysis mode (assuming it's available in global scope or config)
    # If not available, default to 'investment'
    analysis_mode = globals().get('ANALYSIS_OBJECTIVE', 'investment')
    
    # Prepare data only for specified companies
    plot_data = []
    outlier_companies = []
    
    for ticker in companies:
        if ticker not in results.companies:
            continue
            
        analysis = results.companies[ticker]
        if analysis.current_price is None:
            continue
        
        # Use PROJECTED CAGR for growth metric (not historical)
        period = config.PRIMARY_PERIOD
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        fcf_cagr = fcf_proj.annual_growth_median * 100
        rev_cagr = rev_proj.annual_growth_median * 100
        combined_growth = (fcf_cagr * config.FCF_GROWTH_WEIGHT + 
                          rev_cagr * config.REVENUE_GROWTH_WEIGHT)
        
        # Check for outliers (using projected growth)
        if abs(combined_growth) > 100:
            outlier_companies.append({
                'ticker': ticker.split(':')[0],
                'growth': combined_growth,
                'iv_ratio': analysis.best_iv_to_price_ratio
            })
            continue  # Skip outliers for main plot
        
        # Get opportunity rank
        opp_rank = np.inf
        if ticker in results.combined_rankings['ticker'].values:
            opp_rank = results.combined_rankings[
                results.combined_rankings['ticker'] == ticker
            ]['opportunity_rank'].values[0]
        
        # Get growth rank
        growth_rank = np.inf
        if ticker in results.growth_rankings['ticker'].values:
            growth_rank = results.growth_rankings[
                results.growth_rankings['ticker'] == ticker
            ]['combined_growth_rank'].values[0]
        
        # Get risk-adjusted rank for growth mode
        risk_adj_rank = np.inf
        if ticker in results.combined_rankings['ticker'].values:
            risk_adj_rank = results.combined_rankings[
                results.combined_rankings['ticker'] == ticker
            ]['risk_adjusted_rank'].values[0]
        
        plot_data.append({
            'ticker': ticker.split(':')[0],
            'growth': combined_growth,
            'growth_rank': growth_rank,
            'iv_ratio': analysis.best_iv_to_price_ratio,
            'market_cap': analysis.company_data.market_cap / 1e9,
            'stability': analysis.growth_stability_score,
            'opportunity_rank': opp_rank,
            'risk_adjusted_rank': risk_adj_rank,
            'has_divergence': analysis.growth_divergence > config.GROWTH_DIVERGENCE_THRESHOLD
        })
    
    if plot_data:
        df = pd.DataFrame(plot_data)
        logger.info(f"Projected growth range: {df['growth'].min():.1f}% to {df['growth'].max():.1f}%")
        logger.info(f"IV/Price ratio range: {df['iv_ratio'].min():.2f} to {df['iv_ratio'].max():.2f}")
        
        # Size based on market cap (scaled for visibility)
        sizes = 50 + (df['market_cap'] / df['market_cap'].max()) * 300
        
        # Color based on appropriate rank for the analysis mode
        if analysis_mode == 'growth':
            # For growth mode, color by risk-adjusted rank
            colors = df['risk_adjusted_rank']
            color_label = 'Risk-Adjusted Growth Rank (Lower is Better)'
        else:
            # For investment mode, color by opportunity rank
            colors = df['opportunity_rank']
            color_label = 'Investment Opportunity Rank (Lower is Better)'
        
        # Create scatter plot with growth rank vs IV ratio
        scatter = ax.scatter(df['growth_rank'], df['iv_ratio'],
                            s=sizes,
                            c=colors,
                            cmap='viridis_r',  # Reversed so better ranks are brighter
                            alpha=0.8,
                            edgecolors='black',
                            linewidth=1)
        
        # Add labels
        for _, row in df.iterrows():
            # Add divergence warning symbol if applicable
            label = row['ticker']
            if row['has_divergence']:
                label += ' ⚠️'
            
            ax.annotate(label, 
                       (row['growth_rank'], row['iv_ratio']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, rotation=270, labelpad=20)
    
    # Add outlier annotation if any
    if outlier_companies:
        outlier_text = f"Outliers (>100% Projected CAGR):\n"
        for out in outlier_companies[:5]:
            growth_text = f">100%" if out['growth'] > 100 else f"<-100%"
            outlier_text += f"{out['ticker']}: {growth_text}, IV/P={out['iv_ratio']:.1f}\n"
        if len(outlier_companies) > 5:
            outlier_text += f"... and {len(outlier_companies)-5} more"
        
        ax.text(0.02, 0.98, outlier_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Add reference lines
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Fair Value (IV = Price)')
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.3, label='2x Undervalued')
            
    if plot_data:
        median_growth_rank = df['growth_rank'].median()
        ax.axvline(x=median_growth_rank, color='black', linestyle='--', alpha=0.3)
        
        # Work directly with rank values
        min_rank = df['growth_rank'].min()  # Best growth
        max_rank = df['growth_rank'].max()  # Worst growth
        
        # Calculate centers based on data, not axis limits
        x_right = (min_rank + median_growth_rank) / 2  # Good growth quadrants
        x_left = (median_growth_rank + max_rank) / 2   # Poor growth quadrants
        
        # Y positions remain the same
        y_bottom = 0.65  # Overvalued zone
        y_top = 2.0      # Undervalued zone

        # Quadrant labels based on analysis mode
        if analysis_mode == 'growth':
            # Growth mode: emphasize growth quality and stability
            ax.text(x_right, y_top, 'High Growth\nUndervalued\n(Monitor for Entry)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.text(x_left, y_top, 'Low Growth\nUndervalued\n(Value Traps?)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.text(x_right, y_bottom, 'High Growth\nFairly Valued\n(Growth Stars)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            ax.text(x_left, y_bottom, 'Low Growth\nOvervalued\n(Avoid)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        else:
            # Investment mode: emphasize immediate opportunity
            ax.text(x_right, y_top, 'High Growth\nGood Value\n(Best Buys)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.text(x_left, y_top, 'Low Growth\nGood Value\n(Deep Value)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.text(x_right, y_bottom, 'High Growth\nPoor Value\n(Too Expensive)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            ax.text(x_left, y_bottom, 'Low Growth\nPoor Value\n(Avoid)', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Growth Rank (Lower = Better Growth)', fontsize=14)
    ax.set_ylabel('Intrinsic Value / Current Price Ratio', fontsize=14)
    
    # Title based on analysis mode
    if analysis_mode == 'growth':
        title = f'Growth Potential vs. Valuation Analysis\n'
        subtitle = f'Colored by Risk-Adjusted Growth Quality | Based on {config.PRIMARY_PERIOD}-Year Projected CAGR'
    else:
        title = f'Investment Opportunity Matrix\n'
        subtitle = f'Colored by Overall Investment Rank | Based on {config.PRIMARY_PERIOD}-Year Projected CAGR'
    
    ax.set_title(title + subtitle, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    if plot_data:
        ax.set_xlim(0, max(df['growth_rank']) * 1.1)
    ax.set_ylim(0, 4)  # Cap at 4x undervalued for clarity
    
    # Invert x-axis so better growth rank is on the right
    ax.invert_xaxis()
    
    # Add size legend
    sizes = [0.1, 1, 5, 10]
    labels = ['$0.1B', '$1B', '$5B', '$10B']
    markers = []
    for size, label in zip(sizes, labels):
        markers.append(ax.scatter([], [], s=50 + (size/10) * 300, c='gray', 
                                 edgecolors='black', linewidth=1))
    
    legend1 = ax.legend(markers, labels, title='Market Cap', loc='upper left', 
                       frameon=True, fancybox=True)
    ax.add_artist(legend1)
    
    # Add main legend
    ax.legend(loc='lower right')
    
    # Add note about divergence warnings
    if any(row.get('has_divergence', False) for row in plot_data):
        ax.text(0.98, 0.02, '* indicates FCF/Revenue growth divergence > 10%', 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig


def create_financial_health_dashboard(company: CompanyAnalysis, ax: Optional[plt.Axes] = None, config: Optional[AnalysisConfig] = None):
    """
    Create financial health metrics dashboard.
    
    Args:
        company: Company analysis object
        ax: Optional matplotlib axes
        config: Analysis configuration
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.axis('off')
    
    # Key metrics
    debt_cash = company.company_data.lt_debt / company.company_data.cash_and_equiv if company.company_data.cash_and_equiv > 0 else np.inf
    fcf_yield = company.company_data.fcf / company.company_data.market_cap
    op_margin = company.company_data.operating_income / company.company_data.revenue
    acquirers = company.company_data.enterprise_value / company.company_data.operating_income if company.company_data.operating_income != 0 else np.inf
    
    # Create gauge-style visualization with viridis colors
    metrics = [
        ('Debt/Cash', debt_cash, 2.5, 'lower'),
        ('FCF Yield', fcf_yield * 100, 10, 'higher'),
        ('Op Margin', op_margin * 100, 15, 'higher'),
        ("Acquirer's Multiple", acquirers, 15, 'lower')
    ]
    
    # Use viridis colormap for status colors
    good_color = plt.cm.viridis(0.8)
    medium_color = plt.cm.viridis(0.5)
    bad_color = plt.cm.viridis(0.2)
    
    y_pos = 0.8
    for name, value, threshold, better in metrics:
        # Determine color
        if better == 'higher':
            if value >= threshold:
                color = good_color
            elif value >= threshold * 0.7:
                color = medium_color
            else:
                color = bad_color
        else:
            if value <= threshold:
                color = good_color
            elif value <= threshold * 1.3:
                color = medium_color
            else:
                color = bad_color
        
        # Draw metric
        ax.text(0.1, y_pos, name + ':', fontweight='bold', fontsize=12)
        
        if name == "Acquirer's Multiple":
            ax.text(0.6, y_pos, f'{value:.1f}x', fontsize=12, color=color, fontweight='bold')
        elif 'Yield' in name or 'Margin' in name:
            ax.text(0.6, y_pos, f'{value:.1f}%', fontsize=12, color=color, fontweight='bold')
        else:
            ax.text(0.6, y_pos, f'{value:.2f}', fontsize=12, color=color, fontweight='bold')
        
        y_pos -= 0.2
    
    ax.set_title('Financial Health Indicators', fontsize=14, fontweight='bold', 
                pad=20, loc='center')
    
    return ax


def create_ranking_comparison_table(results: AnalysisResults, companies: List[str], config: AnalysisConfig):
    """
    Create formatted ranking comparison table.
    
    Args:
        results: Analysis results
        companies: List of company tickers
        config: Analysis configuration (required)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Prepare data
    rows = []
    for ticker in companies:
        if ticker not in results.companies:
            logger.warning(f"Company {ticker} not found in results for ranking table")
            continue
            
        analysis = results.companies[ticker]
        
        # Get ranks from different DataFrames
        growth_rank = results.growth_rankings[
            results.growth_rankings['ticker'] == ticker
        ]['combined_growth_rank'].values[0] if len(results.growth_rankings[results.growth_rankings['ticker'] == ticker]) > 0 else 999
        
        value_rank = results.value_rankings[
            results.value_rankings['ticker'] == ticker
        ]['value_rank'].values[0] if len(results.value_rankings[results.value_rankings['ticker'] == ticker]) > 0 else 999
        
        opp_rank = results.combined_rankings[
            results.combined_rankings['ticker'] == ticker
        ]['opportunity_rank'].values[0] if len(results.combined_rankings[results.combined_rankings['ticker'] == ticker]) > 0 else 999
        
        # Get growth rates
        period = config.PRIMARY_PERIOD
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        rows.append([
            ticker.split(':')[0],
            f'${analysis.current_price:.2f}',
            f'{fcf_proj.annual_growth_median:.1%}',
            f'{rev_proj.annual_growth_median:.1%}',
            f'{analysis.best_iv_to_price_ratio:.2f}x',
            f'{analysis.growth_stability_score:.2f}',
            f'#{int(growth_rank)}',
            f'#{int(value_rank)}',
            f'#{int(opp_rank)}'
        ])
    
    # Create table
    columns = ['Ticker', 'Price', 'FCF Gr.', 'Rev Gr.', 'IV/Price', 
              'Stability', 'Growth Rk', 'Value Rk', 'Opp. Rk']
    
    table = ax.table(cellText=rows,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header with viridis
    header_color = plt.cm.viridis(0.8)
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code opportunity rank column with viridis gradient
    for i in range(1, len(rows) + 1):
        opp_rank_val = int(rows[i-1][8].replace('#', ''))
        if opp_rank_val <= 5:
            color = plt.cm.viridis(0.9)
        elif opp_rank_val <= 10:
            color = plt.cm.viridis(0.7)
        elif opp_rank_val <= 20:
            color = plt.cm.viridis(0.5)
        else:
            color = plt.cm.viridis(0.3)
        table[(i, 8)].set_facecolor(color)
    
    ax.set_title('Comparative Rankings', fontsize=16, fontweight='bold', pad=20)
    
    return fig


def create_projected_growth_stability_chart(results: AnalysisResults, companies: List[str], config: AnalysisConfig):
    """
    Create scatter plot of projected FCF growth (base case) vs stability.
    
    This chart helps identify companies with the best combination of expected future growth 
    and predictability based on Monte Carlo simulations. Companies in the upper right 
    quadrant are ideal as they combine strong projected growth with stable, predictable patterns.
    
    The growth rates shown are PROJECTED CAGR from the median (base case) scenario
    of the Monte Carlo simulations, NOT historical growth rates.
    
    Args:
        results: Analysis results containing Monte Carlo projections
        companies: List of company tickers (should be top_companies_by_iv)
        config: Analysis configuration
        
    Returns:
        Matplotlib figure showing projected growth vs stability
    """
    logger.info(f"Creating projected growth vs stability chart for {len(companies)} companies")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data only for specified companies
    plot_data = []
    outlier_companies = []
    
    for ticker in companies:
        if ticker not in results.companies:
            continue
            
        analysis = results.companies[ticker]
        if analysis.current_price is None:
            continue
        
        # Get PROJECTED growth rate from base case scenario (median)
        period = config.PRIMARY_PERIOD
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        fcf_growth_annual = fcf_proj.annual_growth_median * 100  # Convert to percentage
        
        # Get growth stability
        stability = analysis.growth_stability_score if hasattr(analysis, 'growth_stability_score') and analysis.growth_stability_score is not None else 0.0
        
        # Check for outliers (extreme projected growth rates that would distort the chart)
        # These are companies where Monte Carlo simulations project extremely high/low growth
        if abs(fcf_growth_annual) > 100:
            outlier_companies.append({
                'ticker': ticker.split(':')[0],
                'fcf_growth': fcf_growth_annual,
                'stability': stability
            })
            continue  # Skip outliers for main plot to maintain readable scale
        
        # Calculate growth quality score (combination of growth and stability)
        # Normalise growth to 0-1 range (assuming -10% to +50% annual growth range for projections)
        growth_norm = (fcf_growth_annual + 10) / 60
        growth_norm = max(0, min(1, growth_norm))  # Clamp to [0, 1]
        
        # Quality score weights stability more heavily
        quality_score = (growth_norm * 0.4 + stability * 0.6)
        
        plot_data.append({
            'ticker': ticker.split(':')[0],
            'fcf_growth': fcf_growth_annual,
            'stability': stability,
            'quality_score': quality_score,
            'market_cap': analysis.company_data.market_cap / 1e9  # In billions
        })
    
    if plot_data:
        df = pd.DataFrame(plot_data)
        logger.info(f"Projected growth range: {df['fcf_growth'].min():.1f}% to {df['fcf_growth'].max():.1f}%")
        logger.info(f"Stability range: {df['stability'].min():.2f} to {df['stability'].max():.2f}")
        
        # ===== CHANGE 1: Add rank calculation =====
        # Calculate growth quality ranks among the companies being plotted
        # Higher quality score = better rank (lower number)
        df['quality_rank'] = df['quality_score'].rank(ascending=False, method='min')
        
        # Size based on market cap (scaled for visibility)
        sizes = 50 + (df['market_cap'] / df['market_cap'].max()) * 300
        
        # ===== CHANGE 2: Update scatter plot to use rank instead of score =====
        # Create scatter plot with viridis colormap
        # Note: Using reversed colormap so better ranks (lower numbers) appear brighter
        scatter = ax.scatter(df['fcf_growth'], df['stability'],
                            s=sizes,
                            c=df['quality_rank'],  # CHANGED from quality_score
                            cmap='viridis_r',       # CHANGED from viridis (reversed)
                            alpha=0.8,
                            edgecolors='black',
                            linewidth=1,
                            vmin=1, vmax=len(df))   # CHANGED from vmin=0, vmax=1
        
        # Add labels
        for _, row in df.iterrows():
            ax.annotate(row['ticker'], 
                       (row['fcf_growth'], row['stability']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # ===== CHANGE 3: Update colorbar label =====
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Growth Quality Rank (Lower is Better)', rotation=270, labelpad=20)  # CHANGED label
        
        # Calculate dynamic quadrant boundaries
        growth_median = df['fcf_growth'].median()
        stability_median = df['stability'].median()
        
        logger.info(f"Quadrant boundaries - Growth: {growth_median:.1f}%, Stability: {stability_median:.2f}")
        
        # Add quadrant lines
        ax.axvline(x=growth_median, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=stability_median, color='black', linestyle='--', alpha=0.3)
        
        # Add quadrant labels
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        x_mid_left = x_range[0] + (growth_median - x_range[0]) / 2
        x_mid_right = growth_median + (x_range[1] - growth_median) / 2
        y_mid_bottom = y_range[0] + (stability_median - y_range[0]) / 2
        y_mid_top = stability_median + (y_range[1] - stability_median) / 2
        
        # Quadrant labels with background boxes
        ax.text(x_mid_left, y_mid_bottom, 'Unstable\nSlow Growth\n(Avoid)', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(x_mid_right, y_mid_bottom, 'Unstable\nFast Growth', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax.text(x_mid_left, y_mid_top, 'Stable\nSlow Growth', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
        ax.text(x_mid_right, y_mid_top, 'Stable\nFast Growth\n(Ideal)', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Add outlier annotation if any
    if outlier_companies:
        outlier_text = "Outliers (>100% Projected CAGR):\n"
        for out in outlier_companies[:5]:
            growth_text = ">100%" if out['fcf_growth'] > 100 else "<-100%"
            outlier_text += f"{out['ticker']}: {growth_text}, Stability={out['stability']:.2f}\n"
        if len(outlier_companies) > 5:
            outlier_text += f"... and {len(outlier_companies)-5} more"
        
        ax.text(0.02, 0.98, outlier_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Formatting
    ax.set_xlabel(f'Projected FCF Growth Rate - Base Case ({config.PRIMARY_PERIOD}-Year CAGR %)', fontsize=14)
    ax.set_ylabel('Growth Stability (Higher = More Stable)', fontsize=14)
    ax.set_title(f'Projected FCF Growth Rate vs. Stability Analysis\n' +
                 f'Based on {config.PRIMARY_PERIOD}-Year Monte Carlo Base Case (Median) Projections\n' +
                 'Size = Market Cap, Color = Growth Quality Rank', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add reference line at 10% growth
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10% Growth Target')
    
    # Add size legend
    sizes = [1, 10, 50]
    labels = ['$1B', '$10B', '$50B']
    markers = []
    for size, label in zip(sizes, labels):
        markers.append(ax.scatter([], [], s=50 + (size/50) * 300, c='gray', 
                                 edgecolors='black', linewidth=1))
    
    legend1 = ax.legend(markers, labels, title='Market Cap', loc='upper left', 
                       frameon=True, fancybox=True)
    ax.add_artist(legend1)
    
    # Add main legend
    ax.legend(loc='lower right')
    
    return fig

def create_risk_adjusted_opportunity_chart(results: AnalysisResults, companies: List[str], config: AnalysisConfig):
    """
    Create scatter plot of risk-adjusted opportunity rankings.
    
    This chart visualises the trade-off between opportunity and risk, helping
    identify companies that offer the best risk-adjusted returns. Companies
    in the lower-left quadrant are ideal (high opportunity, low risk).
    
    Args:
        results: Analysis results
        companies: List of company tickers (watchlist)
        config: Analysis configuration
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Creating risk-adjusted opportunity chart for {len(companies)} companies")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data only for specified companies
    plot_data = []
    outlier_companies = []  # ADD THIS
    
    for ticker in companies:
        if ticker not in results.companies:
            continue
            
        analysis = results.companies[ticker]
        if analysis.current_price is None:
            continue
        
        # Get rankings and scores from combined_rankings
        if ticker not in results.combined_rankings['ticker'].values:
            continue
        
        row_data = results.combined_rankings[results.combined_rankings['ticker'] == ticker].iloc[0]
        
        # Extract key metrics
        opp_rank = row_data['opportunity_rank']
        risk_adj_rank = row_data['risk_adjusted_rank']
        volatility = row_data['avg_volatility']
        
        # Get growth and value metrics for quadrant determination
        growth_score = row_data['combined_growth']
        iv_ratio = row_data['best_iv_ratio']
        
        # ADD OUTLIER CHECK HERE
        # Check if this company has extreme growth
        fcf_cagr = analysis.company_data.historical_fcf_cagr * 100
        rev_cagr = analysis.company_data.historical_revenue_cagr * 100
        
        if abs(fcf_cagr) > 100 or abs(rev_cagr) > 100:
            outlier_companies.append({
                'ticker': ticker.split(':')[0],
                'fcf_cagr': fcf_cagr,
                'rev_cagr': rev_cagr,
                'volatility': volatility
            })
            continue  # Skip outliers for main plot
        
        plot_data.append({
            'ticker': ticker.split(':')[0],
            'opportunity_rank': opp_rank,
            'risk_adjusted_rank': risk_adj_rank,
            'volatility': volatility,
            'growth_score': growth_score,
            'iv_ratio': iv_ratio,
            'market_cap': analysis.company_data.market_cap / 1e9
        })
    
    if plot_data:
        df = pd.DataFrame(plot_data)
        
        # Normalise ranks for color mapping
        norm = plt.Normalize(vmin=1, vmax=df['risk_adjusted_rank'].max())
        
        # Size based on market cap
        sizes = 50 + (df['market_cap'] / df['market_cap'].max()) * 300
        
        scatter = ax.scatter(df['opportunity_rank'], df['volatility'],
                            s=sizes,
                            c=df['risk_adjusted_rank'],
                            cmap='viridis_r',  # Reversed so better ranks are brighter
                            norm=norm,
                            alpha=0.8,
                            edgecolors='black',
                            linewidth=1)
        
        # Add labels
        for _, row in df.iterrows():
            ax.annotate(row['ticker'], 
                       (row['opportunity_rank'], row['volatility']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Risk-Adjusted Rank (Lower is Better)', rotation=270, labelpad=20)
        
        # Add quadrant lines based on medians
        opp_median = df['opportunity_rank'].median()
        vol_median = df['volatility'].median()
        
        ax.axvline(x=opp_median, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=vol_median, color='black', linestyle='--', alpha=0.3)
        
        # Get axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Calculate quadrant centers
        # Remember: x-axis is inverted, so lower rank (better) is on the right
        left_center = (x_max + opp_median) / 2  # Left side (worse opportunity)
        right_center = (x_min + opp_median) / 2  # Right side (better opportunity)
        bottom_center = (y_min + vol_median) / 2
        top_center = (vol_median + y_max) / 2
        
        # Add quadrant labels with correct positioning for inverted x-axis
        ax.text(right_center, top_center, 'High Opportunity\nHigh Risk', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax.text(left_center, top_center, 'Low Opportunity\nHigh Risk\n(Avoid)', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(right_center, bottom_center, 'High Opportunity\nLow Risk\n(Ideal)', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(left_center, bottom_center, 'Low Opportunity\nLow Risk', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Opportunity Rank (Lower = Better)', fontsize=14)
    ax.set_ylabel('Average Volatility (Annual %)', fontsize=14)
    ax.set_title('Risk-Adjusted Opportunity Analysis\nSize = Market Cap, Color = Risk-Adjusted Rank', 
                fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Invert x-axis so better ranks (lower numbers) are on the right
    ax.invert_xaxis()
    
    # Add reference line for target volatility
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% Volatility Target')
    
    # Add size legend
    sizes = [0.1, 1, 10, 50]
    labels = ['$0.1B', '$1B', '$10B', '$50B']
    markers = []
    for size, label in zip(sizes, labels):
        markers.append(ax.scatter([], [], s=50 + (size/50) * 300, c='gray', 
                                 edgecolors='black', linewidth=1))
    
    legend1 = ax.legend(markers, labels, title='Market Cap', loc='upper left', 
                       frameon=True, fancybox=True)
    ax.add_artist(legend1)
    
    # Add main legend
    ax.legend(loc='lower right')
    
    return fig

import matplotlib.pyplot as plt
import numpy as np
from typing import List

# Assuming the necessary data structures (AnalysisResults, AnalysisConfig) are defined elsewhere

def create_comprehensive_rankings_table(results: "AnalysisResults", companies: List[str], config: "AnalysisConfig"):
    """
    Create comprehensive table showing various rankings and growth metrics.
    
    This table provides a complete view of all ranking factors, helping
    understand why companies are ranked as they are and making it easy
    to compare across different metrics.
    
    Now shows PROJECTED growth rates from Monte Carlo simulations.
    
    Args:
        results: Analysis results
        companies: List of company tickers (original list is ignored in favor of sorting by rank)
        config: Analysis configuration
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, 'Comprehensive Company Rankings and Growth Metrics', 
            ha='center', va='top', fontsize=18, fontweight='bold', 
            transform=ax.transAxes)
    
    # Subtitle with methodology note
    ax.text(0.5, 0.94, f'Rankings combine growth potential, valuation, and risk metrics | Growth rates shown as {config.PRIMARY_PERIOD}-year projected CAGR', 
            ha='center', va='top', fontsize=11, style='italic',
            transform=ax.transAxes)
 
    # 1. Sort tickers by risk-adjusted rank instead of using the passed 'companies' list order
    sorted_rankings = results.combined_rankings.sort_values(by='risk_adjusted_rank')
    sorted_tickers = sorted_rankings['ticker'].tolist()
    
    # Prepare data
    rows = []
    # Loop through the new sorted list of tickers
    for ticker in sorted_tickers[:25]:  # Limit to 25 for readability
        if ticker not in results.companies:
            continue
        
        analysis = results.companies[ticker]
        if ticker not in results.combined_rankings['ticker'].values:
            continue
        
        # Get all ranking data
        combined_row = results.combined_rankings[results.combined_rankings['ticker'] == ticker].iloc[0]
        
        # Get growth rank from growth_rankings DataFrame
        growth_rank = 999  # Default if not found
        if ticker in results.growth_rankings['ticker'].values:
            growth_rank = results.growth_rankings[results.growth_rankings['ticker'] == ticker]['combined_growth_rank'].values[0]
        
        # Get value rank from value_rankings DataFrame
        value_rank = 999  # Default if not found
        if ticker in results.value_rankings['ticker'].values:
            value_rank = results.value_rankings[results.value_rankings['ticker'] == ticker]['value_rank'].values[0]
        
        # Get PROJECTED CAGR values (not historical)
        period = config.PRIMARY_PERIOD
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        fcf_cagr = fcf_proj.annual_growth_median * 100
        rev_cagr = rev_proj.annual_growth_median * 100
        
        # Get scenario ranges
        fcf_conservative = analysis.get_projection(period, 'fcf', 'bottom').annual_growth_median * 100
        fcf_optimistic = analysis.get_projection(period, 'fcf', 'top').annual_growth_median * 100
        
        # Format CAGR for display
        fcf_cagr_text = f'{fcf_cagr:.1f}%'
        rev_cagr_text = f'{rev_cagr:.1f}%'
        
        # Calculate annual volatility from quarterly variance
        fcf_vol = np.sqrt(analysis.company_data.fcf_growth_variance) * 2 * 100
        rev_vol = np.sqrt(analysis.company_data.revenue_growth_variance) * 2 * 100
        
        # 2. Append row data in the new specified order
        rows.append([
            ticker.split(':')[0],                               # Ticker
            f'#{int(combined_row["risk_adjusted_rank"])}',       # Risk-Adj Rank
            f'#{int(combined_row["opportunity_rank"])}',         # Opp Rank
            f'#{int(value_rank)}',                               # Value Rank
            f'#{int(growth_rank)}',                              # Growth Rank
            fcf_cagr_text,                                      # Projected FCF CAGR
            f'{fcf_conservative:.0f}%-{fcf_optimistic:.0f}%',    # FCF Range
            rev_cagr_text,                                      # Projected Rev CAGR
            f'{fcf_vol:.0f}%',                                   # FCF Vol
            f'{rev_vol:.0f}%',                                   # Rev Vol
            f'{analysis.best_iv_to_price_ratio:.2f}x',           # IV/Price
            f'{analysis.growth_stability_score:.2f}',           # Stability
            f'${analysis.current_price:.2f}',                   # Price
            f'${analysis.company_data.market_cap/1e9:.2f}B'     # Market Cap
        ])
    
    # 3. Define columns in the new specified order
    columns = ['Ticker', 'Risk-Adj\nRank', 'Opp.\nRank', 'Value\nRank', 'Growth\nRank', 
               f'{period}yr FCF\nCAGR', 'FCF Range\n(C-O)', f'{period}yr Rev\nCAGR', 
               'FCF\nVol', 'Rev\nVol', 'IV/Price', 'Stability', 'Price', 'Mkt Cap']
    
    # Create the table
    table = ax.table(cellText=rows,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0.05, 1, 0.85])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header with viridis
    header_color = plt.cm.viridis(0.8)
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 4. Color code cells based on performance, using UPDATED column indices
    for i in range(1, len(rows) + 1):
        # Color risk-adjusted rank column (now at index 1)
        risk_rank_val = int(rows[i-1][1].replace('#', ''))
        if risk_rank_val <= 5:
            color = plt.cm.viridis(0.9)
        elif risk_rank_val <= 10:
            color = plt.cm.viridis(0.7)
        elif risk_rank_val <= 20:
            color = plt.cm.viridis(0.5)
        else:
            color = plt.cm.viridis(0.3)
        table[(i, 1)].set_facecolor(color)
        
        # Highlight high volatility cells (>30%)
        # FCF Vol is now at index 8
        fcf_vol = float(rows[i-1][8].replace('%', ''))
        if fcf_vol > 30:
            table[(i, 8)].set_facecolor('#FFEB3B')
        
        # Rev Vol is now at index 9
        rev_vol = float(rows[i-1][9].replace('%', ''))
        if rev_vol > 30:
            table[(i, 9)].set_facecolor('#FFEB3B')
        
        # Highlight high growth (>20% CAGR)
        # FCF CAGR is now at index 5
        fcf_cagr = float(rows[i-1][5].replace('%', ''))
        if fcf_cagr > 20:
            table[(i, 5)].set_facecolor('#E8F5E9')
            
    # Add legend
    ax.text(0.05, 0.02, f'Note: Growth rates are {config.PRIMARY_PERIOD}-year projected CAGR from Monte Carlo simulation | C-O = Conservative to Optimistic range | Volatility cells highlighted in yellow if >30% | Rankings: lower numbers = better', 
            transform=ax.transAxes, fontsize=9, style='italic')
    
    return fig

def create_acquirers_multiple_analysis(results: AnalysisResults, companies: List[str], config: AnalysisConfig):
    """
    Create acquirer's multiple analysis chart showing historical vs current.
    
    Args:
        results: Analysis results
        companies: List of company tickers to analyze
        config: Analysis configuration
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data
    am_data = []
    for ticker in companies[:20]:  # Limit to 20 companies for readability
        if ticker not in results.companies:
            continue
            
        analysis = results.companies[ticker]
        
        # Historical AM
        hist_am = (analysis.company_data.enterprise_value / 
                  analysis.company_data.operating_income 
                  if analysis.company_data.operating_income != 0 else np.nan)
        
        # Current AM (from Yahoo Finance data)
        current_am = analysis.current_acquirers_multiple if analysis.current_acquirers_multiple else np.nan
        
        if not np.isnan(hist_am) and not np.isnan(current_am) and hist_am > 0:
            am_data.append({
                'ticker': ticker.split(':')[0],
                'historical_am': hist_am,
                'current_am': current_am,
                'pct_change': ((current_am - hist_am) / hist_am * 100)
            })
    
    if not am_data:
        ax.text(0.5, 0.5, 'No valid Acquirer\'s Multiple data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    am_df = pd.DataFrame(am_data)
    positions = np.arange(len(am_df))
    
    # Create violin plots for historical distribution
    # Since we only have point estimates, create synthetic distributions
    violin_data = []
    for _, row in am_df.iterrows():
        # Create synthetic distribution with 20% coefficient of variation
        hist_val = row['historical_am']
        synthetic = np.random.normal(hist_val, hist_val * 0.2, 1000)
        synthetic = synthetic[synthetic > 0]  # Remove negative values
        violin_data.append(synthetic)
    
    violin_parts = ax.violinplot(violin_data, positions=positions, widths=0.6,
                                showmeans=True, showmedians=False, showextrema=True)
    
    # Color violins with viridis
    for pc in violin_parts['bodies']:
        pc.set_facecolor(plt.cm.viridis(0.3))
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
    
    # Plot current AM as red dots
    ax.scatter(positions, am_df['current_am'], color='red', s=100, zorder=5, 
              label='Current AM', edgecolors='black', linewidth=1)
    
    # Add value labels
    for i, row in am_df.iterrows():
        ax.text(i, row['current_am'] + max(am_df['current_am']) * 0.02, 
                f"{row['current_am']:.1f}", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add percentage change labels below
    y_min = ax.get_ylim()[0]
    for i, row in am_df.iterrows():
        color = 'green' if row['pct_change'] < 0 else 'red'
        ax.text(i, y_min - (ax.get_ylim()[1] - y_min) * 0.05, 
                f"{row['pct_change']:.1f}%", 
                ha='center', va='top', fontsize=9, color=color, fontweight='bold')
    
    # Add reference line
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Reference AM = 10')
    
    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(am_df['ticker'], rotation=45, ha='right')
    ax.set_ylabel("Acquirer's Multiple (EV/EBIT)", fontsize=12)
    ax.set_title("Historical (hist_) vs Current (rt_acquirers_multiple) Acquirer's Multiple", 
                fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    
    # Adjust y-axis
    y_max = max(100, am_df[['historical_am', 'current_am']].max().max() * 1.1)
    ax.set_ylim(0, y_max)
    
    # Add margin at bottom for percentage labels
    ax.margins(y=0.15)
    
    return fig

# Helper functions for scenario comparison
def _create_dcf_waterfall(ax, company: CompanyAnalysis, config: AnalysisConfig):
    """Create DCF waterfall chart."""
    period = config.PRIMARY_PERIOD
    
    iv_key = ('fcf', 'median')
    if period not in company.intrinsic_values:
        logger.error(f"No intrinsic values for period {period}")
        raise KeyError(f"No intrinsic values for period {period}")
    if iv_key not in company.intrinsic_values[period]:
        logger.error(f"No FCF median intrinsic value for period {period}")
        raise KeyError(f"No FCF median intrinsic value for period {period}")
    
    iv = company.intrinsic_values[period][iv_key]
    
    # Components
    components = ['Current FCF', 'Growth Value', 'Terminal Value', 'Discount', 
                 'Safety Margin', 'Per Share']
    values = [
        company.company_data.fcf / company.company_data.shares_diluted,
        sum(iv.projected_cash_flows) / company.company_data.shares_diluted - 
        company.company_data.fcf / company.company_data.shares_diluted,
        iv.terminal_value / company.company_data.shares_diluted,
        -iv.present_value * 0.2 / company.company_data.shares_diluted,  # Approximation
        -iv.present_value * config.MARGIN_OF_SAFETY / company.company_data.shares_diluted,
        0  # Final value
    ]
    
    # Calculate cumulative
    cumulative = []
    total = 0
    for i, val in enumerate(values[:-1]):
        cumulative.append(total)
        total += val
    cumulative.append(iv.intrinsic_value_per_share)
    
    # Plot with viridis colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(components)))
    
    for i, (comp, val, cum, color) in enumerate(zip(components, values, cumulative, colors)):
        if i < len(components) - 1:
            ax.bar(i, abs(val), bottom=min(cum, cum + val), 
                  color=color, alpha=0.7, edgecolor='black')
        else:
            ax.bar(i, cum, color=color, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.set_ylabel('Value per Share ($)')
    ax.set_title('DCF Value Build-up')
    ax.grid(True, axis='y', alpha=0.3)


def _create_sensitivity_analysis(ax, company: CompanyAnalysis, config: AnalysisConfig):
    """Create sensitivity analysis chart."""
    # Variables to test
    discount_rates = np.linspace(0.08, 0.12, 5)
    growth_adjustments = np.linspace(-0.02, 0.02, 5)
    
    # Base case IV
    period = config.PRIMARY_PERIOD
    base_iv = company.get_intrinsic_value(period, 'fcf', 'median')
    
    # Get base growth rate (CAGR)
    fcf_proj = company.get_projection(period, 'fcf', 'median')
    base_growth = fcf_proj.annual_growth_median
    
    # Create sensitivity matrix
    sensitivity = np.zeros((5, 5))
    for i, dr in enumerate(discount_rates):
        for j, ga in enumerate(growth_adjustments):
            # Approximate sensitivity
            sensitivity[i, j] = base_iv * (config.DISCOUNT_RATE / dr) * \
                               (1 + ga / base_growth if base_growth != 0 else 1)
    
    # Plot heatmap with viridis
    im = ax.imshow(sensitivity, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'{g:+.0%}' for g in growth_adjustments])
    ax.set_yticklabels([f'{d:.0%}' for d in discount_rates])
    
    ax.set_xlabel('Growth Adjustment')
    ax.set_ylabel('Discount Rate')
    ax.set_title('Valuation Sensitivity')
    
    # Add value annotations
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'${sensitivity[i, j]:.0f}',
                   ha='center', va='center', fontsize=8)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)


def _create_growth_trajectories(ax, company: CompanyAnalysis, config: AnalysisConfig):
    """Create growth trajectory comparison."""
    years = np.arange(0, config.PRIMARY_PERIOD + 1)
    period = config.PRIMARY_PERIOD
    
    # Get projections for different scenarios
    fcf_proj_bottom = company.get_projection(period, 'fcf', 'bottom')
    fcf_proj_median = company.get_projection(period, 'fcf', 'median')
    fcf_proj_top = company.get_projection(period, 'fcf', 'top')
    
    # Plot different scenarios with viridis colors
    colors = plt.cm.viridis([0.2, 0.5, 0.8])
    scenarios = [
        ('Conservative', fcf_proj_bottom.annual_growth_ci_bottom, colors[0]),
        ('Base', fcf_proj_median.annual_growth_median, colors[1]),
        ('Optimistic', fcf_proj_top.annual_growth_ci_top, colors[2])
    ]
    
    fcf_current = company.company_data.fcf / 1e6
    
    for name, rate, color in scenarios:
        # Use CAGR for annual projections
        values = fcf_current * (1 + rate) ** years
        ax.plot(years, values, marker='o', color=color, 
               label=f'{name} (CAGR: {rate:.1%})', linewidth=2)
    
    ax.set_xlabel('Years')
    ax.set_ylabel('FCF ($M)')
    ax.set_title('FCF Growth Scenarios (CAGR)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _create_risk_return_profile(ax, company: CompanyAnalysis, config: AnalysisConfig):
    """Create risk-return profile visualization."""
    # Scenarios
    scenarios = ['Conservative', 'Base', 'Optimistic']
    returns = []
    risks = []
    
    period = config.PRIMARY_PERIOD
    
    for scenario, scenario_key in zip(['bottom', 'median', 'top'], ['bottom', 'median', 'top']):
        iv = company.get_intrinsic_value(period, 'fcf', scenario_key)
        if company.current_price:
            returns.append((iv / company.current_price - 1) * 100)
        else:
            returns.append(0)
    
    # Get growth projections for risk calculation
    fcf_proj_bottom = company.get_projection(period, 'fcf', 'bottom')
    fcf_proj_median = company.get_projection(period, 'fcf', 'median')
    fcf_proj_top = company.get_projection(period, 'fcf', 'top')
    
    # Use growth variance as risk proxy (convert to annual)
    risks = [
        np.sqrt(fcf_proj_bottom.annual_growth_ci_bottom ** 2) * 100,
        np.sqrt(fcf_proj_median.annual_growth_median ** 2) * 100,
        np.sqrt(fcf_proj_top.annual_growth_ci_top ** 2) * 100
    ]
    
    # Plot with viridis colors
    colors = plt.cm.viridis([0.2, 0.5, 0.8])
    for i, (scenario, ret, risk, color) in enumerate(zip(scenarios, returns, risks, colors)):
        ax.scatter(risk, ret, s=200, color=color, edgecolor='black', 
                  linewidth=2, label=scenario)
    
    ax.set_xlabel('Risk (Growth Volatility %)')
    ax.set_ylabel('Expected Return (%)')
    ax.set_title('Risk-Return Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add reference line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    """
FILE: report_visualizations.py
LOCATION: reports/report_visualizations.py
PURPOSE: Add these enhanced growth comparison functions to the existing report_visualizations.py file

These functions create separate historical and projected growth comparison charts
with proper labeling and error bars for uncertainty visualization.

FUNCTIONS TO ADD:
1. create_growth_comparison_historical() - Shows actual past performance
2. create_growth_comparison_projected() - Shows expected future performance with uncertainty
3. create_growth_comparison_both() - Creates both charts together

NOTE: The original create_growth_comparison() function should be updated to call
create_growth_comparison_historical() for backward compatibility.
"""

# ============================================================================
# FUNCTION 1: Historical Growth Comparison
# ============================================================================

def create_growth_comparison_historical(results: AnalysisResults, save_path: Optional[str] = None) -> plt.Figure:
    """
    FILE: report_visualizations.py
    FUNCTION: create_growth_comparison_historical
    
    Create bar chart comparing HISTORICAL FCF and revenue growth rates (CAGR).
    
    This chart shows actual past performance, helping identify companies with
    historically consistent growth across both free cash flow and revenue.
    
    Data source: Historical CAGR calculated from 20 quarters of actual data
    Divergence: Based on historical differences between FCF and revenue growth
    
    Args:
        results: Analysis results containing company data
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure showing historical growth comparison
    """
    # Initialize containers for chart data
    companies = []
    fcf_growth = []
    revenue_growth = []
    divergence_flags = []
    outlier_companies = []
    
    # Process top 20 watchlist companies
    for ticker in results.watchlist[:20]:
        if ticker not in results.companies:
            continue
            
        analysis = results.companies[ticker]
        
        # Extract historical CAGR values (already calculated during data loading)
        fcf_cagr = analysis.company_data.historical_fcf_cagr * 100  # Convert to percentage
        rev_cagr = analysis.company_data.historical_revenue_cagr * 100
        
        # Identify outliers (extreme growth rates that would distort the chart scale)
        if abs(fcf_cagr) > 100 or abs(rev_cagr) > 100:
            outlier_companies.append({
                'ticker': ticker.split(':')[0],
                'fcf_cagr': fcf_cagr,
                'rev_cagr': rev_cagr
            })
            continue  # Skip outliers for main chart
        
        companies.append(ticker.split(':')[0])
        fcf_growth.append(fcf_cagr)
        revenue_growth.append(rev_cagr)
        
        # Calculate HISTORICAL divergence between FCF and revenue growth
        historical_divergence = abs(fcf_cagr - rev_cagr)
        divergence_flags.append(historical_divergence > results.config.GROWTH_DIVERGENCE_THRESHOLD * 100)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if companies:  # Only plot if we have non-outlier companies
        x = np.arange(len(companies))
        width = 0.35
        
        # Use viridis colormap for consistency
        colors = plt.cm.viridis([0.3, 0.7])
        
        # Create grouped bar chart
        fcf_bars = ax.bar(x - width/2, fcf_growth, width, label='FCF CAGR', color=colors[0], alpha=0.8)
        rev_bars = ax.bar(x + width/2, revenue_growth, width, label='Revenue CAGR', color=colors[1], alpha=0.8)
        
        # Add value labels on each bar
        for bars in [fcf_bars, rev_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # Add divergence warnings where FCF and revenue growth differ significantly
        for i, (has_divergence, company) in enumerate(zip(divergence_flags, companies)):
            if has_divergence:
                # Calculate and display the actual divergence percentage
                div_value = abs(fcf_growth[i] - revenue_growth[i])
                ax.text(i, max(fcf_growth[i], revenue_growth[i]) + 5, 
                       f'*\n{div_value:.0f}%',
                       ha='center', va='bottom', fontsize=10, color='red')
        
        # Add reference line at 10% growth (common target)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% Target')
        
        # Chart formatting
        ax.set_xlabel('Company', fontsize=12)
        ax.set_ylabel('Historical Growth Rate (%)', fontsize=12)
        ax.set_title('HISTORICAL FCF vs Revenue Growth Rates (Point-to-Point CAGR)\n'
                     '* indicates historical growth divergence > 10%', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(companies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    # Add outlier annotation box if any companies had extreme growth
    if outlier_companies:
        outlier_text = "Outliers (>100% CAGR):\n"
        for out in outlier_companies[:5]:
            fcf_text = f">{100 if out['fcf_cagr'] > 0 else -100}%" if abs(out['fcf_cagr']) > 100 else f"{out['fcf_cagr']:.0f}%"
            rev_text = f">{100 if out['rev_cagr'] > 0 else -100}%" if abs(out['rev_cagr']) > 100 else f"{out['rev_cagr']:.0f}%"
            outlier_text += f"{out['ticker']}: FCF {fcf_text}, Rev {rev_text}\n"
        if len(outlier_companies) > 5:
            outlier_text += f"... and {len(outlier_companies)-5} more"
        
        ax.text(0.98, 0.98, outlier_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# FUNCTION 2: Projected Growth Comparison with Uncertainty
# ============================================================================

def create_growth_comparison_projected(results: AnalysisResults, save_path: Optional[str] = None) -> plt.Figure:
    """
    FILE: report_visualizations.py
    FUNCTION: create_growth_comparison_projected
    
    Create bar chart comparing PROJECTED FCF and revenue growth rates with uncertainty bands.
    
    This chart shows expected future performance based on Monte Carlo simulations,
    with error bars indicating the range between conservative (25th percentile)
    and optimistic (75th percentile) scenarios.
    
    IMPORTANT: The error bars show the actual 25th and 75th percentile growth rates
    from the Monte Carlo simulation, NOT confidence intervals around those percentiles.
    
    Data source: Monte Carlo projections based on historical growth patterns
    Divergence: Based on projected differences in the base case (median) scenario
    
    Args:
        results: Analysis results containing company projections
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure showing projected growth with uncertainty
    """
    # Initialize containers for chart data
    companies = []
    fcf_growth_median = []
    fcf_growth_p25 = []  # 25th percentile (conservative scenario)
    fcf_growth_p75 = []  # 75th percentile (optimistic scenario)
    revenue_growth_median = []
    revenue_growth_p25 = []
    revenue_growth_p75 = []
    divergence_flags = []
    outlier_companies = []
    
    period = results.config.PRIMARY_PERIOD
    
    # Process top 20 watchlist companies
    for ticker in results.watchlist[:20]:
        if ticker not in results.companies:
            continue
            
        analysis = results.companies[ticker]
        
        # Extract projected growth rates from Monte Carlo scenarios
        # CRITICAL: Use annual_growth_median from each scenario, NOT the CI bounds
        try:
            # FCF projections - get the median growth rate from each scenario
            fcf_proj_bottom = analysis.get_projection(period, 'fcf', 'bottom')
            fcf_proj_median = analysis.get_projection(period, 'fcf', 'median')
            fcf_proj_top = analysis.get_projection(period, 'fcf', 'top')
            
            # Revenue projections - get the median growth rate from each scenario
            rev_proj_bottom = analysis.get_projection(period, 'revenue', 'bottom')
            rev_proj_median = analysis.get_projection(period, 'revenue', 'median')
            rev_proj_top = analysis.get_projection(period, 'revenue', 'top')
            
            # Extract the actual percentile values
            # Bottom scenario = 25th percentile, Median = 50th, Top = 75th
            fcf_p25 = fcf_proj_bottom.annual_growth_median * 100  # 25th percentile
            fcf_p50 = fcf_proj_median.annual_growth_median * 100  # 50th percentile (median)
            fcf_p75 = fcf_proj_top.annual_growth_median * 100     # 75th percentile
            
            rev_p25 = rev_proj_bottom.annual_growth_median * 100  # 25th percentile
            rev_p50 = rev_proj_median.annual_growth_median * 100  # 50th percentile (median)
            rev_p75 = rev_proj_top.annual_growth_median * 100     # 75th percentile
            
            # Check for outliers (based on median values)
            if abs(fcf_p50) > 100 or abs(rev_p50) > 100:
                outlier_companies.append({
                    'ticker': ticker.split(':')[0],
                    'fcf_median': fcf_p50,
                    'rev_median': rev_p50
                })
                continue  # Skip outliers for main chart
            
            companies.append(ticker.split(':')[0])
            fcf_growth_median.append(fcf_p50)
            fcf_growth_p25.append(fcf_p25)
            fcf_growth_p75.append(fcf_p75)
            revenue_growth_median.append(rev_p50)
            revenue_growth_p25.append(rev_p25)
            revenue_growth_p75.append(rev_p75)
            
            # Calculate PROJECTED divergence using base case (median) values
            projected_divergence = abs(fcf_p50 - rev_p50)
            divergence_flags.append(projected_divergence > results.config.GROWTH_DIVERGENCE_THRESHOLD * 100)
            
        except Exception as e:
            logger.warning(f"Could not get projections for {ticker}: {e}")
            continue
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if companies:  # Only plot if we have non-outlier companies
        x = np.arange(len(companies))
        width = 0.35
        
        # Use viridis colormap for consistency
        colors = plt.cm.viridis([0.3, 0.7])
        
        # Create bars showing median (base case) values
        fcf_bars = ax.bar(x - width/2, fcf_growth_median, width, 
                          label='FCF Projected (Base)', color=colors[0], alpha=0.8)
        rev_bars = ax.bar(x + width/2, revenue_growth_median, width, 
                          label='Revenue Projected (Base)', color=colors[1], alpha=0.8)
        
        # Calculate error bar sizes (distance from median to 25th/75th percentiles)
        fcf_err_lower = [med - p25 for med, p25 in zip(fcf_growth_median, fcf_growth_p25)]
        fcf_err_upper = [p75 - med for med, p75 in zip(fcf_growth_median, fcf_growth_p75)]
        rev_err_lower = [med - p25 for med, p25 in zip(revenue_growth_median, revenue_growth_p25)]
        rev_err_upper = [p75 - med for med, p75 in zip(revenue_growth_median, revenue_growth_p75)]
        
        # Add error bars showing uncertainty range
        ax.errorbar(x - width/2, fcf_growth_median, 
                   yerr=[fcf_err_lower, fcf_err_upper],
                   fmt='none', ecolor='black', capsize=3, alpha=0.7)
        ax.errorbar(x + width/2, revenue_growth_median, 
                   yerr=[rev_err_lower, rev_err_upper],
                   fmt='none', ecolor='black', capsize=3, alpha=0.7)
        
        # Add value labels on bars (median values)
        for bars in [fcf_bars, rev_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # Add divergence warnings for PROJECTED divergence
        for i, (has_divergence, company) in enumerate(zip(divergence_flags, companies)):
            if has_divergence:
                # Calculate and display the actual divergence percentage
                div_value = abs(fcf_growth_median[i] - revenue_growth_median[i])
                # Position warning above the higher error bar
                y_pos = max(fcf_growth_median[i] + fcf_err_upper[i], 
                           revenue_growth_median[i] + rev_err_upper[i]) + 5
                ax.text(i, y_pos, 
                       f'*\n{div_value:.0f}%',
                       ha='center', va='bottom', fontsize=10, color='red')
        
        # Add reference line at 10% growth
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% Target')
        
        # Chart formatting
        ax.set_xlabel('Company', fontsize=12)
        ax.set_ylabel('Projected Annual Growth Rate (%)', fontsize=12)
        ax.set_title(f'PROJECTED FCF vs Revenue Growth Rates ({period} Year Monte Carlo)\n'
                     'Error bars show Conservative (25th) to Optimistic (75th) percentile | '
                     '* indicates projected divergence > 10%', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(companies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add explanatory note about scenarios
        note_text = ("Base Case = 50th percentile (median) | Error bars = 25th-75th percentile range\n"
                    "From 10,000 Monte Carlo simulations per company")
        ax.text(0.5, -0.25, note_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, style='italic')
    
    # Add outlier annotation box if any companies had extreme projected growth
    if outlier_companies:
        outlier_text = "Outliers (>100% projected growth):\n"
        for out in outlier_companies[:5]:
            fcf_text = f">{100 if out['fcf_median'] > 0 else -100}%" if abs(out['fcf_median']) > 100 else f"{out['fcf_median']:.0f}%"
            rev_text = f">{100 if out['rev_median'] > 0 else -100}%" if abs(out['rev_median']) > 100 else f"{out['rev_median']:.0f}%"
            outlier_text += f"{out['ticker']}: FCF {fcf_text}, Rev {rev_text}\n"
        if len(outlier_companies) > 5:
            outlier_text += f"... and {len(outlier_companies)-5} more"
        
        ax.text(0.98, 0.98, outlier_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# FUNCTION 3: Create Both Charts Together
# ============================================================================

def create_growth_comparison_both(results: AnalysisResults) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create both historical and projected growth comparison charts.
    
    This function generates two complementary views:
    1. Historical performance (what actually happened)
    2. Projected performance with uncertainty (what we expect)
    
    Args:
        results: Analysis results
        
    Returns:
        Tuple of (historical figure, projected figure)
    """
    # Create historical chart
    fig_hist = create_growth_comparison_historical(results)
    
    # Create projected chart
    fig_proj = create_growth_comparison_projected(results)
    
    return fig_hist, fig_proj

def create_valuation_upside_comparison(results: AnalysisResults, companies: List[str], config: AnalysisConfig) -> plt.Figure:
    """
    Create FCF valuation upside/downside comparison for multiple companies.
    
    This visualization shows the percentage difference between current price and
    FCF-based intrinsic values across three scenarios (Conservative, Base, Optimistic)
    for all specified companies. Companies are displayed in the order provided,
    making it easy to identify which offer the most attractive valuations.
    
    The chart uses horizontal bars extending right for upside (undervalued) and
    left for downside (overvalued), with the zero line as a clear reference point.
    
    Args:
        results: Analysis results containing intrinsic values
        companies: List of company tickers to display (in desired order)
        config: Analysis configuration
        
    Returns:
        Matplotlib figure showing upside/downside comparison
    """
    logger.info(f"Creating FCF valuation upside/downside comparison for {len(companies)} companies")
    
    # Prepare data for all companies
    valuation_data = []
    companies_with_issues = []
    
    period = config.PRIMARY_PERIOD
    
    for ticker in companies:
        if ticker not in results.companies:
            companies_with_issues.append((ticker, "Not found in results"))
            continue
            
        analysis = results.companies[ticker]
        
        # Skip if no current price
        if analysis.current_price is None or analysis.current_price <= 0:
            companies_with_issues.append((ticker, "No current price"))
            continue
        
        # Calculate upside/downside for each scenario
        try:
            conservative_iv = analysis.get_intrinsic_value(period, 'fcf', 'bottom')
            base_iv = analysis.get_intrinsic_value(period, 'fcf', 'median')
            optimistic_iv = analysis.get_intrinsic_value(period, 'fcf', 'top')
            
            conservative_upside = (conservative_iv / analysis.current_price - 1) * 100
            base_upside = (base_iv / analysis.current_price - 1) * 100
            optimistic_upside = (optimistic_iv / analysis.current_price - 1) * 100
            
            valuation_data.append({
                'ticker': ticker.split(':')[0],  # Remove country code for display
                'conservative': conservative_upside,
                'base': base_upside,
                'optimistic': optimistic_upside,
                'current_price': analysis.current_price,
                'order': len(valuation_data)  # Preserve input order
            })
            
        except Exception as e:
            logger.warning(f"Could not calculate valuations for {ticker}: {e}")
            companies_with_issues.append((ticker, str(e)))
            continue
    
    if companies_with_issues:
        logger.info(f"{len(companies_with_issues)} companies excluded due to data issues")
    
    if not valuation_data:
        logger.error("No valid valuation data to display")
        raise ValueError("No companies with valid valuation data")
    
    # Create DataFrame preserving original order
    val_df = pd.DataFrame(valuation_data)
    val_df = val_df.sort_values('order')  # Maintain input order
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, max(8, len(val_df) * 0.5)))
    
    # Set up positions for bars
    y_positions = np.arange(len(val_df))
    bar_height = 0.25
    
    # Use viridis colors for scenarios
    colors = plt.cm.viridis([0.2, 0.5, 0.8])
    
    # Create horizontal bars for each scenario
    conservative_bars = ax.barh(y_positions - bar_height, val_df['conservative'], 
                               bar_height, label='Conservative (25th %ile)', 
                               color=colors[0], alpha=0.8)
    
    base_bars = ax.barh(y_positions, val_df['base'], 
                       bar_height, label='Base Case (50th %ile)', 
                       color=colors[1], alpha=0.8)
    
    optimistic_bars = ax.barh(y_positions + bar_height, val_df['optimistic'], 
                             bar_height, label='Optimistic (75th %ile)', 
                             color=colors[2], alpha=0.8)
    
    # Add value labels on bars
    def add_bar_labels(bars, values):
        for bar, value in zip(bars, values):
            width = bar.get_width()
            label_x = width + 2 if width > 0 else width - 2
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{value:.0f}%', 
                   ha='left' if width > 0 else 'right', 
                   va='center', fontsize=8)
    
    add_bar_labels(conservative_bars, val_df['conservative'])
    add_bar_labels(base_bars, val_df['base'])
    add_bar_labels(optimistic_bars, val_df['optimistic'])
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add shaded regions
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.05, color='green', label='Undervalued')
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.05, color='red', label='Overvalued')
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(val_df['ticker'])
    ax.set_xlabel('Upside/Downside from Current Price (%)', fontsize=12)
    ax.set_title(f'FCF Valuation Upside/Downside Analysis ({period} Year DCF)\n'
                 f'Based on {config.SIMULATION_REPLICATES:,} Monte Carlo Simulations',
                 fontsize=14, fontweight='bold')
    
    # Position legend to avoid overlapping with data
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add grid for easier reading
    ax.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add summary statistics as text
    avg_base_upside = val_df['base'].mean()
    companies_undervalued = (val_df['conservative'] > 0).sum()
    
    summary_text = (f"Average Base Case Upside: {avg_base_upside:.1f}%\n"
                   f"Companies undervalued in all scenarios: {companies_undervalued}/{len(val_df)}")
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    return fig