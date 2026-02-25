"""
Core visualization functions for analysis results.

Simple wrapper for summary report generation.
"""
import logging
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline.data_structures import AnalysisResults, CompanyAnalysis
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def create_summary_report(results: AnalysisResults, output_dir: str):
    """
    Create all visualizations and save to output directory.
    
    This function generates a complete set of visualizations that provide
    different perspectives on the investment opportunities identified by
    the analysis. All valuations are FCF-based only. Growth rates are shown as CAGR.
    
    Args:
        results: Analysis results
        output_dir: Directory to save visualizations
    """
    from pathlib import Path
    # Import the actual visualization functions from report_visualizations
    from reports.report_visualizations import (
        create_growth_value_scatter,
        create_growth_comparison_both,
        create_valuation_upside_comparison,
        create_acquirers_multiple_analysis
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Growth-Value Matrix
    logger.info("Creating growth-value matrix")
    fig1 = create_growth_value_scatter(results, results.watchlist, results.config)
    fig1.savefig(output_path / "growth_value_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Growth Comparison - BOTH Historical and Projected
    logger.info("Creating growth comparison charts (historical and projected)")
    fig_hist, fig_proj = create_growth_comparison_both(results)
    
    # Save both charts with descriptive names
    fig_hist.savefig(output_path / "growth_comparison_historical.png", dpi=300, bbox_inches='tight')
    fig_proj.savefig(output_path / "growth_comparison_projected.png", dpi=300, bbox_inches='tight')
    plt.close(fig_hist)
    plt.close(fig_proj)
    
# 3. Valuation upside/downside comparison
    logger.info("Creating valuation upside/downside comparison")
    fig3 = create_valuation_upside_comparison(results, results.watchlist, results.config)
    fig3.savefig(output_path / "valuation_upside_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Acquirer's Multiple Analysis
    logger.info("Creating acquirer's multiple analysis")
    fig4 = create_acquirers_multiple_analysis(results, results.watchlist, results.config)
    fig4.savefig(output_path / "acquirers_multiple_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    logger.info(f"All visualizations saved to {output_path}")