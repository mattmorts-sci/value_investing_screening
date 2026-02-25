"""
Methodology document generator.

Creates detailed documentation of all calculations and assumptions.
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np

from config.user_config import AnalysisConfig


def generate_methodology_document(config: AnalysisConfig, output_path: Path):
    """
    Generate comprehensive methodology documentation PDF.
    
    Args:
        config: Analysis configuration
        output_path: Path for output PDF
    """
    with PdfPages(str(output_path)) as pdf:
        # Title page
        _add_methodology_title(pdf)
        
        # Data structure
        _add_data_structure_section(pdf, config)
        
        # Filtering methodology
        _add_filtering_methodology(pdf, config)
        
        # Growth calculation
        _add_growth_calculation_methodology(pdf, config)
        
        # Intrinsic value calculation
        _add_intrinsic_value_methodology(pdf, config)
        
        # Ranking methodology
        _add_ranking_methodology(pdf, config)
        
        # Example calculations
        _add_example_calculations(pdf, config)


def _add_methodology_title(pdf: PdfPages):
    """Add methodology document title page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.7, 'Intrinsic Value Analysis\nMethodology Documentation',
            ha='center', va='center', fontsize=22, fontweight='bold')
    
    ax.text(0.5, 0.5, 'Complete Guide to Calculations and Assumptions',
            ha='center', va='center', fontsize=14)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_data_structure_section(pdf: PdfPages, config: AnalysisConfig):
    """Document data structure and sources."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = f"""
DATA STRUCTURE AND SOURCES

1. Historical Data Files:
   • Metrics file: metrics-{config.COUNTRY}-FQ-{config.COLLECTION_DATE}.json
   • Growth file: growth-{config.COUNTRY}-FQ-{config.COLLECTION_DATE}.json
   • Period: Quarterly (FQ)
   • History: {config.DATA_LENGTH} quarters

2. Key Metrics Extracted:
   
   Financial Metrics (at quarter {config.DATA_LENGTH - 1}):
   • Free Cash Flow (FCF): Total cash generated from operations
   • Revenue: Total company revenue
   • Operating Income: Earnings before interest and taxes
   • Market Capitalization: Share price × shares outstanding
   • Long-term Debt: Total long-term obligations
   • Cash and Equivalents: Liquid assets
   • Shares Diluted: Total shares including dilutive securities
   • Enterprise Value: Market cap + debt - cash
   
   Growth Statistics (calculated from historical data):
   • FCF Growth Mean: Average quarterly FCF growth rate
   • FCF Growth Variance: Volatility of FCF growth
   • Revenue Growth Mean: Average quarterly revenue growth
   • Revenue Growth Variance: Volatility of revenue growth
   
   Historical CAGR (NEW):
   • FCF CAGR: Point-to-point calculation from first to last valid value
   • Revenue CAGR: Point-to-point calculation from first to last valid value
   • Used for display and ranking purposes

3. Calculated Metrics:
   • Debt/Cash Ratio = Long-term Debt ÷ Cash and Equivalents
   • FCF per Share = FCF ÷ Shares Diluted
   • Acquirer's Multiple = Enterprise Value ÷ Operating Income
   • FCF to Market Cap = FCF ÷ Market Capitalization

4. Real-time Data (Yahoo Finance):
   • Current stock price
   • Current market capitalization
   • Current financial metrics (when available)
"""
    
    ax.text(0.05, 0.95, content, ha='left', va='top', fontsize=9,
            family='monospace', wrap=True)
    
    ax.set_title('Section 1: Data Structure', fontsize=16, fontweight='bold',
                loc='left', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_filtering_methodology(pdf: PdfPages, config: AnalysisConfig):
    """Document filtering methodology."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = f"""
FILTERING METHODOLOGY

Companies must pass ALL filters to be included in analysis:

1. Negative FCF Filter:
   • Requirement: FCF > $0
   • Rationale: Companies burning cash are not suitable for value investing
   • Removes: Cash-burning companies

2. Financial Consistency Filter:
   • Requirement: Operating Income ≤ Revenue
   • Rationale: Data quality check - operating income cannot exceed revenue
   • Removes: Companies with data anomalies

3. Market Capitalization Filter:
   • Requirement: Market Cap ≥ ${config.MIN_MARKET_CAP:,.0f}
   • Rationale: Ensures sufficient liquidity and stability
   • Removes: Micro-cap and illiquid stocks

4. Debt-to-Cash Ratio Filter:
   • Requirement: Debt/Cash Ratio ≤ {config.MAX_DEBT_TO_CASH_RATIO}
   • Rationale: Ensures financial stability
   • Removes: Overleveraged companies
   • Special case: If Cash = 0, ratio = infinity (filtered out)

5. Special Handling for Owned Companies:
   • Mode: {config.ANALYSIS_MODE}
   • If mode = 'owned': Owned companies bypass filters but are flagged
   • This allows analysis of existing holdings regardless of criteria

Filter Application Order:
1. Negative FCF → 2. Financial Consistency → 3. Market Cap → 4. Debt/Cash

Each filter records which companies are removed and why, enabling
transparency in the selection process.
"""
    
    ax.text(0.05, 0.95, content, ha='left', va='top', fontsize=9,
            family='monospace', wrap=True)
    
    ax.set_title('Section 2: Filtering Methodology', fontsize=16, fontweight='bold',
                loc='left', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_growth_calculation_methodology(pdf: PdfPages, config: AnalysisConfig):
    """Document growth calculation methodology."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = f"""
GROWTH CALCULATION METHODOLOGY

Two Separate Growth Calculations:

1. Historical CAGR (for display and ranking):
   • Point-to-point calculation from historical data
   • Formula: CAGR = (Last Value / First Value)^(1/periods) - 1
   • Converted from quarterly to annual: Annual = (1 + Quarterly)^4 - 1
   • Used for: Rankings, displays, and growth comparisons
   • Advantages: Simple, interpretable, not affected by volatility

2. Monte Carlo Simulation (for projections):
   • Simulation Parameters:
     - Simulations per metric: {config.SIMULATION_REPLICATES:,}
     - Projection periods: {config.PROJECTION_PERIODS} years
     - Quarters per year: {config.QUARTERS_PER_YEAR}
   
   • Growth Rate Sampling:
     For each quarter, sample growth rate from:
     Growth ~ Normal(μ, σ²)
     Where:
     - μ = Historical growth mean
     - σ² = Historical growth variance
   
   • Growth Constraints:
     a) Absolute Limits:
        - Maximum quarterly growth: {config.MAX_QUARTERLY_GROWTH:.0%}
        - Minimum quarterly growth: {config.MIN_QUARTERLY_GROWTH:.0%}
     
     b) Size-based Constraints:
        - Small companies (FCF/Market Cap < 0.1%): Factor = 0.8
        - Medium companies (0.1% - 1%): Factor = 0.5-0.9
        - Large companies (> 1%): Factor = 0.4-1.0
     
     c) Time Decay for High Growth:
        - If quarterly growth > 50%: Apply decay
        - Decay factor = 0.8^(years elapsed)

3. Intrinsic Value Scenarios:
   • Conservative (25th percentile growth)
   • Base (50th percentile growth)
   • Optimistic (75th percentile growth)
"""
    
    ax.text(0.05, 0.95, content, ha='left', va='top', fontsize=9,
            family='monospace', wrap=True)
    
    ax.set_title('Section 3: Growth Calculation', fontsize=16, fontweight='bold',
                loc='left', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_intrinsic_value_methodology(pdf: PdfPages, config: AnalysisConfig):
    """Document intrinsic value calculation methodology."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = f"""
INTRINSIC VALUE CALCULATION (DCF MODEL)

1. Discounted Cash Flow Formula:
   
   IV = Σ(CFₜ / (1 + r)ᵗ) + TV / (1 + r)ⁿ
   
   Where:
   • CFₜ = Cash flow in year t
   • r = Discount rate = {config.DISCOUNT_RATE:.1%}
   • TV = Terminal value
   • n = Projection period = {config.PRIMARY_PERIOD} years

2. Cash Flow Projections:
   • Start with current FCF (Free Cash Flow only)
   • Apply growth rates from Monte Carlo simulation
   • Convert quarterly projections to annual cash flows
   • Annual CF = Sum of 4 quarterly cash flows

3. Terminal Value Calculation:
   
   TV = CFₙ × (1 + g) / (r - g)
   
   Where:
   • CFₙ = Final year cash flow
   • g = Terminal growth rate = {config.TERMINAL_GROWTH_RATE:.1%}
   • r = Discount rate = {config.DISCOUNT_RATE:.1%}

4. Present Value Calculation:
   • Discount each annual cash flow to present
   • Discount terminal value to present
   • Sum all discounted values

5. Margin of Safety:
   
   Final IV = Present Value × (1 - Margin of Safety)
   Margin of Safety = {config.MARGIN_OF_SAFETY:.0%}
   
   This provides a conservative estimate

6. Per Share Calculation:
   
   IV per Share = Total IV / Shares Diluted

7. Important Note:
   • Only FCF-based valuations are calculated
   • Revenue is tracked for growth analysis but NOT used for valuation
   • This is theoretically correct as only actual cash flows should be valued
"""
    
    ax.text(0.05, 0.95, content, ha='left', va='top', fontsize=9,
            family='monospace', wrap=True)
    
    ax.set_title('Section 4: Intrinsic Value Calculation', fontsize=16, fontweight='bold',
                loc='left', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_ranking_methodology(pdf: PdfPages, config: AnalysisConfig):
    """Document ranking methodology."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = f"""
RANKING AND SELECTION METHODOLOGY

1. Primary Ranking: Best IV/Price Ratio
   • All companies ranked by FCF-based IV/Price ratio
   • Higher ratio = more undervalued = better rank
   • This becomes the primary selection criterion

2. Additional Ranking Components:

   a) Growth Rank:
      • Based on historical FCF and Revenue CAGR
      • Combined growth = FCF CAGR × 0.7 + Revenue CAGR × 0.3
      • Higher growth = better rank
   
   b) Stability Rank:
      • Growth stability score = 1 / (1 + coefficient of variation)
      • Higher stability = better rank
      • Measures consistency of growth paths
   
   c) Weighted Rank:
      • Debt/Cash penalty weight: {config.DC_WEIGHT}
      • Market Cap penalty weight: {config.MC_WEIGHT}
      • Growth penalty weight: {config.GROWTH_WEIGHT}
      • Lower total penalty = better rank

3. Risk-Adjusted Ranking:
   
   • Incorporates volatility from quarterly growth variance
   • Annual volatility = Quarterly std dev × 2
   • Risk-adjusted score = Opportunity (60%) + Volatility penalty (30%) + Stability (10%)
   • Favours consistent performers over volatile high-growth

4. Final Selection Process:
   
   Step 1: Rank all companies by best IV/Price ratio
   Step 2: Select top N companies for detailed analysis
   Step 3: Apply consistent selection across all visualizations
   Step 4: Flag outliers (>100% CAGR) but don't exclude from ranking

5. Outlier Handling:
   
   • Companies with extreme CAGR (>100%) are:
     - Included in rankings and analysis
     - Excluded from visualization scales
     - Listed separately as outliers
   • This prevents scale distortion while maintaining transparency
"""
    
    ax.text(0.05, 0.95, content, ha='left', va='top', fontsize=9,
            family='monospace', wrap=True)
    
    ax.set_title('Section 5: Ranking Methodology', fontsize=16, fontweight='bold',
                loc='left', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_example_calculations(pdf: PdfPages, config: AnalysisConfig):
    """Add example calculations."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = f"""
EXAMPLE CALCULATIONS

Example Company: ABC Corp
• Historical FCF: Q1=$80M, Q20=$150M
• Current FCF: $100M
• Shares: 50M
• Historical FCF growth mean: 5% quarterly
• Historical FCF growth variance: 0.01

Step 1: Historical CAGR Calculation
• Quarterly CAGR = (150/80)^(1/19) - 1 = 3.36%
• Annual CAGR = (1.0336)^4 - 1 = 14.2%
• This is used for ranking and display

Step 2: Growth Projection (5-year, median scenario)
• Uses Monte Carlo with quarterly mean/variance
• Median annual growth from simulation ≈ 21.55%
• Year 5 FCF = $100M × (1.2155)^5 = $265M

Step 3: DCF Calculation
Year 1: CF = $121.55M, PV = $121.55M / 1.10 = $110.50M
Year 2: CF = $147.76M, PV = $147.76M / 1.10² = $122.12M
Year 3: CF = $179.61M, PV = $179.61M / 1.10³ = $134.90M
Year 4: CF = $218.31M, PV = $218.31M / 1.10⁴ = $149.17M
Year 5: CF = $265.37M, PV = $265.37M / 1.10⁵ = $164.77M

Terminal Value:
TV = $265.37M × 1.01 / (0.10 - 0.01) = $2,981.65M
PV of TV = $2,981.65M / 1.10⁵ = $1,851.00M

Total PV = $681.46M + $1,851.00M = $2,532.46M

Step 4: Apply Margin of Safety
Conservative IV = $2,532.46M × (1 - 0.50) = $1,266.23M

Step 5: Per Share Value
IV per share = $1,266.23M / 50M shares = $25.32

Step 6: IV/Price Ratio
If current price = $15
IV/Price ratio = $25.32 / $15 = 1.69
"""
    
    ax.text(0.05, 0.95, content, ha='left', va='top', fontsize=9,
            family='monospace', wrap=True)
    
    ax.set_title('Section 6: Example Calculations', fontsize=16, fontweight='bold',
                loc='left', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)