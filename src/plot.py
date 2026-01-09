import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List

# Global plot settings - set once
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.autolayout'] = True

def plot_panel(data: pd.DataFrame, x: str, y: str, col: str, 
               title: str, kind: str = 'line', col_wrap: int = 3,
               height: int = 4, aspect: float = 1.2, 
               figsize: tuple = (12, 8)) -> sns.FacetGrid:
    """
    Optimized panel plotting function.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    x, y : str
        Column names for x and y axes
    col : str
        Column to create subplots by
    title : str
        Main title
    kind : str
        Plot type ('line', 'scatter')
    col_wrap : int
        Number of columns before wrapping
    height, aspect : float
        Subplot dimensions
    figsize : tuple
        Overall figure size
    """
    # Convert to categorical if small unique values for memory efficiency
    if data[col].nunique() < 50:
        data = data.copy()
        data[col] = data[col].astype('category')
    
    # Create plot with optimized parameters
    g = sns.relplot(
        data=data, 
        x=x, 
        y=y, 
        col=col,
        kind=kind,
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        facet_kws={'sharex': True, 'sharey': True}  # Optimize axes sharing
    )
    
    # Efficient title setting
    g.fig.suptitle(title, fontsize=14, fontweight='bold')
    g.set_titles("{col_name}")
    g.set_axis_labels(x, y)
    
    # Optimize layout
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return g

def plot_demand_classification(adi_series: pd.Series, cv2_series: pd.Series, 
                               title: str = "Demand Classification") -> plt.Figure:
    """
    Fast ADI vs CV² classification plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Vectorized scatter plot
    scatter = ax.scatter(adi_series, cv2_series, alpha=0.6, s=20, 
                        edgecolors='w', linewidth=0.3)
    
    # Classification boundaries
    ax.axvline(x=1.32, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0.49, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('ADI', fontsize=11)
    ax.set_ylabel('CV²', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_zero_distribution(zero_data: pd.DataFrame, cols: List[str] = None) -> plt.Figure:
    """
    Efficient zero distribution plotting.
    """
    if cols is None:
        cols = ['avg_zero_sequence_length', 'max_zero_sequence_length', 
                'num_zero_sequences', 'zero_ratio']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(cols[:4]):
        if col in zero_data.columns:
            # Efficient histogram with fixed bins
            axes[idx].hist(zero_data[col].dropna(), bins=30, alpha=0.7, 
                          color=sns.color_palette()[idx])
            axes[idx].set_title(col.replace('_', ' ').title())
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_comparison_boxplots(dataframes: List[pd.DataFrame], 
                            labels: List[str], 
                            metric: str = 'adi') -> plt.Figure:
    """
    Fast boxplot comparison across datasets.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = []
    for df, label in zip(dataframes, labels):
        if metric in df.columns:
            plot_data.append(df[metric].dropna())
    
    # Single boxplot call
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
        patch.set_facecolor(color)
    
    ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_time_series(data: pd.DataFrame, x: str, y: str, 
                    group_col: str = None, title: str = "") -> plt.Figure:
    """
    Fast time series plotting for large datasets.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if group_col and group_col in data.columns:
        # Sample groups if too many
        unique_groups = data[group_col].unique()
        if len(unique_groups) > 10:
            unique_groups = np.random.choice(unique_groups, 10, replace=False)
        
        for group in unique_groups:
            subset = data[data[group_col] == group]
            ax.plot(subset[x], subset[y], alpha=0.7, linewidth=1, 
                   label=str(group))
        ax.legend(fontsize=8, loc='best')
    else:
        # Efficient line plot for single series
        ax.plot(data[x], data[y], linewidth=1, alpha=0.8)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    
    # Optimize date formatting if x is datetime
    if pd.api.types.is_datetime64_any_dtype(data[x]):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Quick test
    print("Plotting functions loaded")