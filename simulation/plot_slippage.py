"""
Visualize empirical Uniswap V3 slippage findings.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import loader from analyze_slippage
sys.path.insert(0, os.path.dirname(__file__))
from analyze_slippage import load_all_pools_for_month, POOLS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    months = [
        ('2023_10', 'Oct 2023'),
        ('2024_03', 'Mar 2024'),
        ('2024_06', 'Jun 2024'),
        ('2024_12', 'Dec 2024'),
    ]
    all_dfs = []
    for suffix, label in months:
        print(f"Loading {label}...")
        df = load_all_pools_for_month(base_dir, suffix, label)
        if len(df) > 0:
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def plot_1_size_vs_slippage(df):
    """Fig 1: Trade size → slippage curve (0.05% pools)."""
    df05 = df[df['fee_pct'] == 0.05].copy()

    sizes = np.logspace(1, 7, 40)  # $10 to $10M
    medians, p25s, p75s, p95s, counts = [], [], [], [], []

    for s in sizes:
        lo, hi = s * 0.6, s * 1.4
        sub = df05[(df05['trade_size_usd'] >= lo) & (df05['trade_size_usd'] <= hi)]
        if len(sub) >= 20:
            medians.append(sub['slippage_bps'].median())
            p25s.append(sub['slippage_bps'].quantile(0.25))
            p75s.append(sub['slippage_bps'].quantile(0.75))
            p95s.append(sub['slippage_bps'].quantile(0.95))
            counts.append(len(sub))
        else:
            medians.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)
            p95s.append(np.nan)
            counts.append(0)

    medians = np.array(medians)
    p25s = np.array(p25s)
    p75s = np.array(p75s)
    p95s = np.array(p95s)

    fig, ax = plt.subplots(figsize=(10, 6))

    mask = ~np.isnan(medians)
    ax.fill_between(sizes[mask], p25s[mask], p75s[mask], alpha=0.2, color='#2196F3',
                     label='25th–75th percentile')
    ax.plot(sizes[mask], medians[mask], '-', color='#2196F3', linewidth=2.5,
            label='Median slippage')
    ax.plot(sizes[mask], p95s[mask], '--', color='#F44336', linewidth=1.5,
            label='95th percentile')

    # Fee line
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(15, 5.5, '0.05% pool fee', fontsize=9, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Trade Size (USD)', fontsize=12)
    ax.set_ylabel('Slippage (bps)', fontsize=12)
    ax.set_title('Uniswap V3 Slippage vs Trade Size\n(USDC/WETH & WETH/USDT, 0.05% fee pools, 4 months)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(10, 1e7)
    ax.set_ylim(-5, 80)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}k' if x >= 1e3 else f'${x:.0f}'
    ))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_size_vs_slippage.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_2_curve_slippage_by_month(df):
    """Fig 2: Pure curve slippage (minus fee) by month."""
    df05 = df[df['fee_pct'] == 0.05].copy()

    sizes = np.logspace(3, 7, 30)  # $1k to $10M
    months = sorted(df05['month'].unique())
    colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 6))

    for month, color in zip(months, colors):
        month_df = df05[df05['month'] == month]
        meds = []
        valid_sizes = []
        for s in sizes:
            lo, hi = s * 0.6, s * 1.4
            sub = month_df[(month_df['trade_size_usd'] >= lo) &
                           (month_df['trade_size_usd'] <= hi)]
            if len(sub) >= 20:
                curve_slip = sub['slippage_bps'].median() - 5.0  # subtract fee
                meds.append(curve_slip)
                valid_sizes.append(s)

        if valid_sizes:
            ax.plot(valid_sizes, meds, 'o-', color=color, linewidth=2,
                    markersize=4, label=month)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Trade Size (USD)', fontsize=12)
    ax.set_ylabel('Curve Slippage Beyond Fee (bps)', fontsize=12)
    ax.set_title('Pure AMM Curve Slippage (fee subtracted)\nby Month and Trade Size', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(1e3, 1e7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}k' if x >= 1e3 else f'${x:.0f}'
    ))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_curve_slippage_by_month.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_3_fee_tier_comparison(df):
    """Fig 3: Slippage distribution by fee tier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fee_tiers = sorted(df['fee_pct'].unique())
    tier_labels = [f'{f:.2f}%' for f in fee_tiers]
    colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0']

    box_data = []
    for fee in fee_tiers:
        sub = df[df['fee_pct'] == fee]['slippage_bps']
        # Clip outliers for visualization
        clipped = sub.clip(lower=-50, upper=200)
        box_data.append(clipped.values)

    bp = ax.boxplot(box_data, labels=tier_labels, patch_artist=True,
                     showfliers=False, widths=0.6,
                     medianprops=dict(color='#D32F2F', linewidth=2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add fee reference lines
    for i, fee in enumerate(fee_tiers):
        fee_bps = fee * 100
        ax.hlines(fee_bps, i + 0.7, i + 1.3, colors='red', linestyles='--',
                  linewidth=1, alpha=0.5)

    ax.set_xlabel('Pool Fee Tier', fontsize=12)
    ax.set_ylabel('Realized Slippage (bps)', fontsize=12)
    ax.set_title('Slippage Distribution by Fee Tier\n(red dashes = fee itself)', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_fee_tier_boxplot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_4_usdc_vs_usdt(df):
    """Fig 4: USDC vs USDT slippage at 0.05% fee tier."""
    df05 = df[df['fee_pct'] == 0.05].copy()

    # Separate by stablecoin
    usdc = df05[df05['pair'] == 'USDC/WETH']
    usdt = df05[df05['pair'] == 'WETH/USDT']

    sizes = np.logspace(2, 7, 35)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: median slippage comparison
    for data, label, color in [(usdc, 'USDC/WETH', '#2196F3'),
                                (usdt, 'WETH/USDT', '#FF9800')]:
        meds, valid = [], []
        for s in sizes:
            lo, hi = s * 0.6, s * 1.4
            sub = data[(data['trade_size_usd'] >= lo) & (data['trade_size_usd'] <= hi)]
            if len(sub) >= 20:
                meds.append(sub['slippage_bps'].median())
                valid.append(s)
        if valid:
            ax1.plot(valid, meds, 'o-', color=color, linewidth=2, markersize=3, label=label)

    ax1.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Trade Size (USD)', fontsize=11)
    ax1.set_ylabel('Median Slippage (bps)', fontsize=11)
    ax1.set_title('Median Slippage: USDC vs USDT', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 40)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}k' if x >= 1e3 else f'${x:.0f}'
    ))
    ax1.grid(True, alpha=0.3)

    # Right: P95 slippage comparison
    for data, label, color in [(usdc, 'USDC/WETH', '#2196F3'),
                                (usdt, 'WETH/USDT', '#FF9800')]:
        p95s, valid = [], []
        for s in sizes:
            lo, hi = s * 0.6, s * 1.4
            sub = data[(data['trade_size_usd'] >= lo) & (data['trade_size_usd'] <= hi)]
            if len(sub) >= 20:
                p95s.append(sub['slippage_bps'].quantile(0.95))
                valid.append(s)
        if valid:
            ax2.plot(valid, p95s, 'o-', color=color, linewidth=2, markersize=3, label=label)

    ax2.set_xscale('log')
    ax2.set_xlabel('Trade Size (USD)', fontsize=11)
    ax2.set_ylabel('P95 Slippage (bps)', fontsize=11)
    ax2.set_title('95th Percentile Slippage: USDC vs USDT', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 200)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}k' if x >= 1e3 else f'${x:.0f}'
    ))
    ax2.grid(True, alpha=0.3)

    plt.suptitle('USDC vs USDT Pools (0.05% fee tier, 4 months)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_usdc_vs_usdt.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_5_slippage_histogram(df):
    """Fig 5: Histogram of realized slippage for different size brackets."""
    df05 = df[df['fee_pct'] == 0.05].copy()

    brackets = [
        ('$1k–$10k', 1_000, 10_000, '#4CAF50'),
        ('$100k–$500k', 100_000, 500_000, '#2196F3'),
        ('$500k–$1M', 500_000, 1_000_000, '#F44336'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (label, lo, hi, color) in zip(axes, brackets):
        sub = df05[(df05['trade_size_usd'] >= lo) & (df05['trade_size_usd'] <= hi)]
        clipped = sub['slippage_bps'].clip(lower=-20, upper=60)

        ax.hist(clipped, bins=80, color=color, alpha=0.7, edgecolor='white', linewidth=0.5,
                density=True)
        ax.axvline(x=5.0, color='black', linestyle='--', linewidth=1.5, label='Pool fee (5bp)')
        ax.axvline(x=sub['slippage_bps'].median(), color='red', linestyle='-',
                   linewidth=1.5, label=f'Median ({sub["slippage_bps"].median():.1f}bp)')

        ax.set_xlabel('Slippage (bps)', fontsize=11)
        ax.set_title(f'{label}\n(n={len(sub):,})', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(-20, 60)

    axes[0].set_ylabel('Density', fontsize=11)
    plt.suptitle('Slippage Distribution by Trade Size (0.05% fee pools)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_slippage_histograms.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_6_vol_regime(df):
    """Fig 6: Slippage under different volatility regimes."""
    df05 = df[df['fee_pct'] == 0.05].copy()

    # Compute rolling volatility proxy
    df05 = df05.sort_values('block_number')
    df05['price_change_bps'] = df05['oracle_price'].pct_change().abs() * 10000
    df05['vol_rolling'] = df05['price_change_bps'].rolling(50, min_periods=10).mean()

    # Split into regimes
    vol_q33 = df05['vol_rolling'].quantile(0.33)
    vol_q67 = df05['vol_rolling'].quantile(0.67)

    regimes = {
        'Low vol': df05[df05['vol_rolling'] <= vol_q33],
        'Med vol': df05[(df05['vol_rolling'] > vol_q33) & (df05['vol_rolling'] <= vol_q67)],
        'High vol': df05[df05['vol_rolling'] > vol_q67],
    }
    colors_map = {'Low vol': '#4CAF50', 'Med vol': '#FF9800', 'High vol': '#F44336'}

    sizes = np.logspace(3, 6.5, 25)

    fig, ax = plt.subplots(figsize=(10, 6))

    for regime_name, regime_df in regimes.items():
        meds, valid = [], []
        for s in sizes:
            lo, hi = s * 0.6, s * 1.4
            sub = regime_df[(regime_df['trade_size_usd'] >= lo) &
                            (regime_df['trade_size_usd'] <= hi)]
            if len(sub) >= 20:
                meds.append(sub['slippage_bps'].median())
                valid.append(s)
        if valid:
            ax.plot(valid, meds, 'o-', color=colors_map[regime_name],
                    linewidth=2, markersize=4, label=regime_name)

    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Trade Size (USD)', fontsize=12)
    ax.set_ylabel('Median Slippage (bps)', fontsize=12)
    ax.set_title('Slippage by Volatility Regime\n(0.05% fee pools)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 50)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}k' if x >= 1e3 else f'${x:.0f}'
    ))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_vol_regime_slippage.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} swaps\n")

    print("Generating figures...")
    plot_1_size_vs_slippage(df)
    plot_2_curve_slippage_by_month(df)
    plot_3_fee_tier_comparison(df)
    plot_4_usdc_vs_usdt(df)
    plot_5_slippage_histogram(df)
    plot_6_vol_regime(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
