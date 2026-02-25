"""
Factor contribution analysis for company scores.

Analyzes which factors (debt/cash, market cap, growth) drive company rankings.
"""
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def calculate_factor_contributions(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage contribution of each factor to total score.
    
    Args:
        scored_df: DataFrame with penalty scores
        
    Returns:
        DataFrame with factor contribution percentages
    """
    factor_df = pd.DataFrame(index=scored_df.index)
    
    # Copy penalty columns
    penalty_cols = ['dc_penalty', 'mc_penalty', 'growth_penalty', 'total_penalty']
    for col in penalty_cols:
        factor_df[col] = scored_df[col]
    
    # Calculate percentages
    for penalty in ['dc_penalty', 'mc_penalty', 'growth_penalty']:
        # Handle case where total penalty is 0 or very small
        factor_df[f'{penalty}_pct'] = np.where(
            factor_df['total_penalty'] > 0.001,
            (factor_df[penalty] / factor_df['total_penalty']) * 100,
            0.0
        )
    
    # Determine primary factor
    factor_cols = ['dc_penalty', 'mc_penalty', 'growth_penalty']
    factor_df['primary_factor'] = factor_df[factor_cols].idxmax(axis=1)
    factor_df['primary_factor_pct'] = factor_df[
        ['dc_penalty_pct', 'mc_penalty_pct', 'growth_penalty_pct']
    ].max(axis=1)
    
    # Add factor labels
    factor_df['primary_factor_label'] = factor_df['primary_factor'].map({
        'dc_penalty': 'Debt/Cash',
        'mc_penalty': 'Market Cap',
        'growth_penalty': 'Growth'
    })
    
    return factor_df


def create_factor_visualization(factor_df: pd.DataFrame, 
                              companies: List[str],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create stacked bar chart showing factor contributions.
    
    Args:
        factor_df: DataFrame with factor contributions
        companies: List of companies to visualize
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Filter to requested companies
    plot_data = factor_df.loc[companies].copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create stacked bars
    x = np.arange(len(companies))
    width = 0.6
    
    dc_pct = plot_data['dc_penalty_pct']
    mc_pct = plot_data['mc_penalty_pct']
    growth_pct = plot_data['growth_penalty_pct']
    
    # Use viridis colors
    colors = plt.cm.viridis([0.2, 0.5, 0.8])
    
    # Create bars
    p1 = ax.bar(x, dc_pct, width, label='Debt/Cash', color=colors[0])
    p2 = ax.bar(x, mc_pct, width, bottom=dc_pct, label='Market Cap', color=colors[1])
    p3 = ax.bar(x, growth_pct, width, bottom=dc_pct+mc_pct, label='Growth', color=colors[2])
    
    # Add percentage labels
    for i, company in enumerate(companies):
        y_pos = 0
        for pct, label in [(dc_pct.iloc[i], 'DC'), 
                          (mc_pct.iloc[i], 'MC'), 
                          (growth_pct.iloc[i], 'G')]:
            if pct >= 10:
                ax.text(i, y_pos + pct/2, f'{pct:.0f}%', 
                       ha='center', va='center', color='white', 
                       fontweight='bold', fontsize=10)
            y_pos += pct
    
    # Formatting
    ax.set_ylabel('Contribution to Total Score (%)', fontsize=12)
    ax.set_title('Factor Contributions to Company Scores', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_factor_heatmap(factor_df: pd.DataFrame,
                         top_n: int = 30,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing factor contributions for top companies.
    
    Args:
        factor_df: DataFrame with factor contributions
        top_n: Number of top companies to show
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Select top companies
    top_companies = factor_df.nsmallest(top_n, 'total_penalty')
    
    # Prepare data for heatmap
    heatmap_data = top_companies[['dc_penalty_pct', 'mc_penalty_pct', 'growth_penalty_pct']].T
    heatmap_data.index = ['Debt/Cash', 'Market Cap', 'Growth']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create heatmap with viridis
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.0f',
                cmap='viridis',
                cbar_kws={'label': 'Contribution (%)'},
                ax=ax)
    
    ax.set_title(f'Factor Contributions for Top {top_n} Companies', fontsize=14)
    ax.set_xlabel('Company', fontsize=12)
    ax.set_ylabel('Factor', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_factor_dominance(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which factors dominate for different company groups.
    
    Args:
        factor_df: DataFrame with factor contributions
        
    Returns:
        Summary DataFrame of factor dominance
    """
    # Group by primary factor
    factor_groups = factor_df.groupby('primary_factor_label')
    
    summary_data = []
    for factor, group in factor_groups:
        summary_data.append({
            'primary_factor': factor,
            'company_count': len(group),
            'pct_of_total': len(group) / len(factor_df) * 100,
            'avg_contribution': group['primary_factor_pct'].mean(),
            'avg_total_penalty': group['total_penalty'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('company_count', ascending=False)
    
    return summary_df


def identify_balanced_companies(factor_df: pd.DataFrame, 
                              balance_threshold: float = 10.0) -> pd.DataFrame:
    """
    Identify companies with balanced factor contributions.
    
    A company is considered balanced if no single factor contributes
    more than a certain percentage difference from perfect balance (33.33%).
    
    Args:
        factor_df: DataFrame with factor contributions
        balance_threshold: Maximum deviation from 33.33% to be considered balanced
        
    Returns:
        DataFrame of balanced companies
    """
    perfect_balance = 100 / 3  # 33.33%
    
    # Calculate maximum deviation from perfect balance
    factor_df['max_deviation'] = factor_df[
        ['dc_penalty_pct', 'mc_penalty_pct', 'growth_penalty_pct']
    ].apply(lambda row: max(abs(row - perfect_balance)), axis=1)
    
    # Identify balanced companies
    balanced = factor_df[factor_df['max_deviation'] <= balance_threshold].copy()
    balanced['balance_score'] = 100 - balanced['max_deviation']
    
    return balanced.sort_values('balance_score', ascending=False)