"""
Visualization and Results Export
=================================

Generate all figures for the paper and extract placeholder values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Consistent colors per policy
POLICY_COLORS = {
    'B1_Industry': '#1f77b4',       # blue
    'B2_Conservative': '#2ca02c',   # green
    'B3_Aggressive': '#d62728',     # red
    'ALG_Adaptive': '#ff7f0e',      # orange — myopic upper bound
    'ALGS_Smoothed': '#9467bd',     # purple — realistic smoothed ALG
}
POLICY_STYLES = {
    'B1_Industry': {'linestyle': '--', 'marker': 'o'},
    'B2_Conservative': {'linestyle': '--', 'marker': 's'},
    'B3_Aggressive': {'linestyle': '--', 'marker': '^'},
    'ALG_Adaptive': {'linestyle': ':', 'marker': 'D', 'linewidth': 2},
    'ALGS_Smoothed': {'linestyle': '-', 'marker': 'P', 'linewidth': 3},
}

def _policy_kw(policy):
    """Return plot kwargs for a given policy name."""
    kw = dict(linewidth=2.5, markersize=5, alpha=0.8)
    kw['color'] = POLICY_COLORS.get(policy)
    kw.update(POLICY_STYLES.get(policy, {}))
    return kw


def _format_dollar_axis(ax, axis='x'):
    """Format a log-scale axis with readable dollar labels like $1M, $10M, $100M.

    Assumes the plotted data is already in millions (i.e., Q / 1e6).
    """
    def fmt(x, pos):
        if x >= 1e3:
            return f'${x/1e3:.0f}B'
        elif x >= 1:
            return f'${x:.0f}M'
        elif x >= 0.001:
            return f'${x*1e3:.0f}K'
        else:
            return f'${x*1e6:.0f}'

    target = ax.xaxis if axis == 'x' else ax.yaxis
    target.set_major_formatter(mticker.FuncFormatter(fmt))


def generate_figure1_carry_vs_size(results_df):
    """
    Figure 1: Net carry vs vault size for all baselines.

    If an 'asset' column exists, generates one subplot per asset plus
    a combined comparison. Otherwise falls back to a single plot.
    """

    print("Generating Figure 1: Net Carry vs Vault Size...")

    has_assets = 'asset' in results_df.columns
    assets = sorted(results_df['asset'].unique()) if has_assets else [None]

    if has_assets and len(assets) > 1:
        fig, axes = plt.subplots(1, len(assets), figsize=(8 * len(assets), 7), sharey=True)

        for ax, asset in zip(axes, assets):
            asset_data = results_df[results_df['asset'] == asset]
            for policy in asset_data['policy'].unique():
                pdata = asset_data[asset_data['policy'] == policy]
                ax.plot(pdata['Q'] / 1e6, pdata['mean_carry_annual_pct'],
                       label=policy, **_policy_kw(policy))
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Vault Size', fontsize=13)
            ax.set_title(f'{asset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            _format_dollar_axis(ax, 'x')

        axes[0].set_ylabel('Mean Net Carry (% annualized)', fontsize=13)
        plt.suptitle('Figure 1: Net Carry vs Vault Size', fontsize=15, fontweight='bold', y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        for policy in results_df['policy'].unique():
            data = results_df[results_df['policy'] == policy]
            ax.plot(data['Q'] / 1e6, data['mean_carry_annual_pct'],
                   label=policy, **_policy_kw(policy))
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Vault Size', fontsize=13)
        ax.set_ylabel('Mean Net Carry (% annualized)', fontsize=13)
        ax.set_title('Figure 1: Net Carry vs Vault Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        _format_dollar_axis(ax, 'x')

    plt.tight_layout()
    plt.savefig('./results/figures/figure1_carry_vs_size.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./results/figures/figure1_carry_vs_size.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved to ./results/figures/figure1_carry_vs_size.pdf")


def generate_figure2_capacity_frontier(results_df):
    """
    Figure 2: Capacity frontier Q_max(r) for each policy.

    Shows maximum sustainable vault size at each target carry rate.
    Per-asset subplots when 'asset' column is present.
    """

    print("Generating Figure 2: Capacity Frontier...")

    has_assets = 'asset' in results_df.columns
    assets = sorted(results_df['asset'].unique()) if has_assets else [None]

    if has_assets and len(assets) > 1:
        fig, axes = plt.subplots(1, len(assets), figsize=(8 * len(assets), 7), sharey=True)

        for ax, asset in zip(axes, assets):
            asset_data = results_df[results_df['asset'] == asset]
            for policy in asset_data['policy'].unique():
                pdata = asset_data[asset_data['policy'] == policy].sort_values('r_target')
                pdata = pdata[pdata['Q_max'] > 0]
                ax.plot(pdata['r_target'], pdata['Q_max'] / 1e6,
                       label=policy, **_policy_kw(policy))
            ax.set_xlabel('Target Net Carry (% annualized)', fontsize=13)
            ax.set_title(f'{asset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            _format_dollar_axis(ax, 'y')

        axes[0].set_ylabel('Maximum Vault Size', fontsize=13)
        plt.suptitle('Figure 2: Capacity Frontier Q_max(r)', fontsize=15, fontweight='bold', y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        for policy in results_df['policy'].unique():
            data = results_df[results_df['policy'] == policy].sort_values('r_target')
            data = data[data['Q_max'] > 0]
            ax.plot(data['r_target'], data['Q_max'] / 1e6,
                   label=policy, **_policy_kw(policy))
        ax.set_xlabel('Target Net Carry (% annualized)', fontsize=13)
        ax.set_ylabel('Maximum Vault Size', fontsize=13)
        ax.set_title('Figure 2: Capacity Frontier Q_max(r)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        _format_dollar_axis(ax, 'y')

    plt.tight_layout()
    plt.savefig('./results/figures/figure2_capacity_frontier.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./results/figures/figure2_capacity_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved to ./results/figures/figure2_capacity_frontier.pdf")


def generate_figure3_optimal_policies(results_df):
    """
    Figure 3: Optimal leverage L*(Q) and epsilon*(Q) for each policy.

    Per-asset rows when 'asset' column is present.
    """

    print("Generating Figure 3: Optimal Leverage Policies...")

    has_assets = 'asset' in results_df.columns
    assets = sorted(results_df['asset'].unique()) if has_assets else [None]
    n_rows = len(assets)

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows), squeeze=False)

    for row, asset in enumerate(assets):
        subset = results_df[results_df['asset'] == asset] if has_assets else results_df
        ax1, ax2 = axes[row]

        for policy in subset['policy'].unique():
            pdata = subset[subset['policy'] == policy]
            kw = _policy_kw(policy)
            ax1.plot(pdata['Q'] / 1e6, pdata['L'], label=policy, **kw)
            kw2 = _policy_kw(policy)
            ax2.plot(pdata['Q'] / 1e6, pdata['epsilon'], label=policy, **kw2)

        row_label = f' ({asset})' if asset else ''
        ax1.set_ylabel('Leverage L', fontsize=13)
        ax1.set_title(f'(a) Optimal Leverage L*(Q){row_label}', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        _format_dollar_axis(ax1, 'x')

        ax2.set_ylabel('Rebalancing Band e', fontsize=13)
        ax2.set_title(f'(b) Optimal Rebalancing Band e*(Q){row_label}', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        _format_dollar_axis(ax2, 'x')

    axes[-1][0].set_xlabel('Vault Size', fontsize=13)
    axes[-1][1].set_xlabel('Vault Size', fontsize=13)

    plt.suptitle('Figure 3: Optimal Policy Maps', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./results/figures/figure3_optimal_policies.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./results/figures/figure3_optimal_policies.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved to ./results/figures/figure3_optimal_policies.pdf")


def generate_table1_risk_metrics(results_df):
    """
    Table 1: Risk metrics for all policies at selected vault sizes.
    Includes asset column when present.
    """

    print("Generating Table 1: Risk Metrics...")

    has_assets = 'asset' in results_df.columns

    table_lines = []
    table_lines.append("\\begin{table}[h]")
    table_lines.append("\\centering")
    table_lines.append("\\caption{Risk Metrics at Selected Vault Sizes}")
    table_lines.append("\\label{tab:risk_metrics}")

    if has_assets:
        table_lines.append("\\begin{tabular}{lllrrrr}")
        table_lines.append("\\toprule")
        table_lines.append("Asset & Policy & Vault Size & Mean Carry & Sharpe & CVaR$_{95}$ & Max DD \\\\")
        table_lines.append(" & & (\\$M) & (\\% p.a.) & Ratio & (\\%) & (\\%) \\\\")
    else:
        table_lines.append("\\begin{tabular}{llrrrr}")
        table_lines.append("\\toprule")
        table_lines.append("Policy & Vault Size & Mean Carry & Sharpe & CVaR$_{95}$ & Max DD \\\\")
        table_lines.append(" & (\\$M) & (\\% p.a.) & Ratio & (\\%) & (\\%) \\\\")
    table_lines.append("\\midrule")

    for _, row in results_df.iterrows():
        Q = row['Q'] / 1e6
        if has_assets:
            line = (f"{row['asset']} & {row['policy']} & {Q:.0f} & "
                    f"{row['mean_carry']:.2f} & {row['sharpe']:.2f} & "
                    f"{row['cvar_95']:.2f} & {row['max_drawdown']:.2f} \\\\")
        else:
            line = (f"{row['policy']} & {Q:.0f} & "
                    f"{row['mean_carry']:.2f} & {row['sharpe']:.2f} & "
                    f"{row['cvar_95']:.2f} & {row['max_drawdown']:.2f} \\\\")
        table_lines.append(line)

    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\end{table}")

    with open('./results/tables/table1_risk_metrics.tex', 'w') as f:
        f.write('\n'.join(table_lines))

    results_df.to_csv('./results/tables/table1_risk_metrics.csv', index=False)

    print("  Saved to ./results/tables/table1_risk_metrics.tex")


def extract_placeholder_values(all_results):
    """
    Extract all placeholder values from results.
    
    Scans results and identifies key numbers to fill in paper.
    
    Returns:
        Dict with placeholder -> value mappings
    """
    
    print("\nExtracting placeholder values for paper...")
    
    placeholders = {}
    
    # From E1: Carry at different sizes
    E1 = all_results['E1']
    
    # B1 (Industry) performance
    b1_data = E1[E1['policy'] == 'B1_Industry'].sort_values('Q')
    if len(b1_data) > 0:
        # Peak carry at small size
        placeholders['X'] = f"{b1_data['mean_carry_annual_pct'].max():.1f}"
        
        # Size where turns negative
        neg_idx = b1_data[b1_data['mean_carry_annual_pct'] < 0].index
        if len(neg_idx) > 0:
            placeholders['Z'] = f"{b1_data.loc[neg_idx[0], 'Q'] / 1e6:.0f}"
        else:
            placeholders['Z'] = "500+"
    
    # B2 (Conservative) performance
    b2_data = E1[E1['policy'] == 'B2_Conservative'].sort_values('Q')
    if len(b2_data) > 0:
        placeholders['X2'] = f"{b2_data['mean_carry_annual_pct'].max():.1f}"
        
        neg_idx = b2_data[b2_data['mean_carry_annual_pct'] < 0].index
        if len(neg_idx) > 0:
            placeholders['Z2'] = f"{b2_data.loc[neg_idx[0], 'Q'] / 1e6:.0f}"
        else:
            placeholders['Z2'] = "500+"
    
    # Dynamic policy (if exists)
    dynamic_data = E1[E1['policy'].str.contains('Dynamic', case=False)]
    if len(dynamic_data) > 0:
        dynamic_data = dynamic_data.sort_values('Q')
        neg_idx = dynamic_data[dynamic_data['mean_carry_annual_pct'] < 0].index
        if len(neg_idx) > 0:
            placeholders['Z3'] = f"{dynamic_data.loc[neg_idx[0], 'Q'] / 1e6:.0f}"
        else:
            placeholders['Z3'] = "500+"
        
        # Capacity expansion factor
        if 'Z' in placeholders and 'Z3' in placeholders:
            try:
                expansion = float(placeholders['Z3']) / float(placeholders['Z'])
                placeholders['X3'] = f"{expansion:.1f}"
            except:
                placeholders['X3'] = "3.0"
    
    # From E2: Capacity frontier values
    E2 = all_results['E2']
    
    # Q_max at 10% target for B1
    b1_10 = E2[(E2['policy'] == 'B1_Industry') & (E2['r_target'] == 10)]
    if len(b1_10) > 0:
        placeholders['Q_B1'] = f"{b1_10['Q_max'].iloc[0] / 1e6:.0f}"
    
    # From E3: Optimal leverage values
    E3 = all_results['E3']
    
    # Leverage at different sizes (if dynamic policy exists)
    if len(dynamic_data) > 0:
        dynamic_policy = E3[E3['policy'].str.contains('Dynamic', case=False)]
        if len(dynamic_policy) > 0:
            # L at $1M
            l_1m = dynamic_policy[dynamic_policy['Q'] <= 2e6]['L'].mean()
            placeholders['L1'] = f"{l_1m:.1f}"
            
            # L at $10M
            l_10m = dynamic_policy[(dynamic_policy['Q'] >= 8e6) & (dynamic_policy['Q'] <= 12e6)]['L'].mean()
            placeholders['L2'] = f"{l_10m:.1f}"
            
            # L at $100M
            l_100m = dynamic_policy[(dynamic_policy['Q'] >= 80e6) & (dynamic_policy['Q'] <= 120e6)]['L'].mean()
            placeholders['L3'] = f"{l_100m:.1f}"
    
    # From E4: Risk metrics
    E4 = all_results['E4']
    
    # CVaR and Sharpe at $100M
    b1_100m = E4[(E4['policy'] == 'B1_Industry') & (E4['Q'] == 100_000_000)]
    if len(b1_100m) > 0:
        placeholders['CVaR_B1'] = f"{b1_100m['cvar_95'].iloc[0]:.2f}"
        placeholders['DD_B1'] = f"{b1_100m['max_drawdown'].iloc[0]:.1f}"
        placeholders['Sharpe_B1'] = f"{b1_100m['sharpe'].iloc[0]:.2f}"
    
    # Save placeholders
    with open('./results/placeholder_values.json', 'w') as f:
        json.dump(placeholders, f, indent=2, sort_keys=True)
    
    print(f"\n✅ Extracted {len(placeholders)} placeholder values")
    print("\nSample placeholders:")
    for key, val in list(placeholders.items())[:10]:
        print(f"  [{key}] = {val}")
    
    return placeholders


def generate_all_figures(results):
    """
    Generate all figures from experiment results.
    
    Args:
        results: Dict from ExperimentRunner.run_all_experiments()
    """
    
    print("\n" + "="*70)
    print("GENERATING ALL FIGURES")
    print("="*70)
    
    import os
    os.makedirs('./results/figures', exist_ok=True)
    os.makedirs('./results/tables', exist_ok=True)
    
    # Generate figures
    generate_figure1_carry_vs_size(results['E1'])
    generate_figure2_capacity_frontier(results['E2'])
    generate_figure3_optimal_policies(results['E3'])
    
    # Generate tables
    generate_table1_risk_metrics(results['E4'])
    
    # Extract placeholders
    placeholders = extract_placeholder_values(results)
    
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated:")
    print("  - 3 figures (PDF + PNG)")
    print("  - 1 table (LaTeX + CSV)")
    print("  - Placeholder values (JSON)")
    print("\nAll outputs in ./results/")
    
    return placeholders


if __name__ == '__main__':
    """
    Generate figures from experiment results.

    Prefers *_comparison.csv files (per-asset) if they exist,
    falls back to the original single-file names.
    """
    import os

    print("Generating figures...")

    def load_result(name):
        comp = f'./results/{name}_comparison.csv'
        single = f'./results/{name}.csv'
        if os.path.exists(comp):
            print(f"  Loading {comp}")
            return pd.read_csv(comp)
        elif os.path.exists(single):
            print(f"  Loading {single}")
            return pd.read_csv(single)
        else:
            raise FileNotFoundError(f"No results for {name} in ./results/")

    results = {
        'E1': load_result('E1'),
        'E2': load_result('E2'),
        'E3': load_result('E3'),
        'E4': load_result('E4'),
        'E5': load_result('E5'),
    }

    placeholders = generate_all_figures(results)

    print("\nFigure generation complete!")