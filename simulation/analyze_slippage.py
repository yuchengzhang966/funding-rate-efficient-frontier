"""
Empirical Slippage Analysis from Real Uniswap V3 Swap Data
============================================================

Slippage = execution_price vs oracle_price

Oracle price = pool price from the PREVIOUS block (before any manipulation
in the current block). This captures both AMM curve slippage and MEV impact.

Loads all tracked pools (WETH/USDC, WETH/USDT at various fee tiers).
"""

import json
import os
import glob
import numpy as np
import pandas as pd

TWO_96 = 2 ** 96

# Pool metadata: pool_id → (token0, token1, fee_bps, decimal0, decimal1)
POOLS = {
    '0x88e6a0': ('USDC', 'WETH', 500, 6, 18),    # USDC/WETH 0.05%
    '0x8ad599': ('USDC', 'WETH', 3000, 6, 18),   # USDC/WETH 0.30%
    '0xe0554a': ('USDC', 'WETH', 100, 6, 18),     # USDC/WETH 0.01%
    '0x7bea39': ('USDC', 'WETH', 10000, 6, 18),   # USDC/WETH 1.00%
    '0x11b815': ('WETH', 'USDT', 500, 18, 6),     # WETH/USDT 0.05%
    '0x4e68cc': ('WETH', 'USDT', 3000, 18, 6),    # WETH/USDT 0.30%
    '0xc7bbec': ('WETH', 'USDT', 100, 18, 6),     # WETH/USDT 0.01%
    '0xc5af84': ('WETH', 'USDT', 10000, 18, 6),   # WETH/USDT 1.00%
}


def sqrtPriceX96_to_eth_price(sqrt_price_x96: float, token0: str,
                                decimal0: int, decimal1: int) -> float:
    """
    Convert sqrtPriceX96 to ETH price in USD-stablecoin terms.

    sqrtPriceX96 = √(token1 / token0) × 2^96  (in raw units)

    Returns price in stablecoin per ETH.
    """
    sqrt_ratio = sqrt_price_x96 / TWO_96
    raw_ratio = sqrt_ratio ** 2  # token1_raw / token0_raw
    decimal_adjustment = 10 ** (decimal0 - decimal1)

    # raw_ratio = token1_raw / token0_raw
    # human_ratio = raw_ratio × 10^(decimal0 - decimal1) = token1 / token0
    human_ratio = raw_ratio * decimal_adjustment

    if token0 in ('USDC', 'USDT'):
        # token0 is stablecoin, token1 is WETH
        # human_ratio = WETH / stablecoin (very small number)
        # ETH price = 1 / human_ratio
        if human_ratio == 0:
            return 0
        return 1.0 / human_ratio
    else:
        # token0 is WETH, token1 is stablecoin
        # human_ratio = stablecoin / WETH = ETH price directly
        return human_ratio


def load_pool_swaps(data_dir: str, pool_prefix: str) -> pd.DataFrame:
    """Load swap data for one pool."""
    meta = POOLS.get(pool_prefix)
    if meta is None:
        return pd.DataFrame()

    token0, token1, fee_bps, dec0, dec1 = meta

    files = glob.glob(os.path.join(data_dir, "*.json"))
    if not files:
        return pd.DataFrame()

    records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    records.append(rec)
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['amount0'] = df['amount0'].astype(float)
    df['amount1'] = df['amount1'].astype(float)
    df['amountUSD'] = df['amountUSD'].astype(float)
    df['sqrtPriceX96'] = df['sqrtPriceX96'].astype(float)
    df['tick'] = df['tick'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    df['block_number'] = df['transaction'].apply(
        lambda x: int(x['blockNumber']) if isinstance(x, dict) else 0
    )

    # Sort by block
    df = df.sort_values(['block_number', 'timestamp']).reset_index(drop=True)

    # Pool price after each swap (in stablecoin per ETH)
    df['pool_price'] = df['sqrtPriceX96'].apply(
        lambda x: sqrtPriceX96_to_eth_price(x, token0, dec0, dec1)
    )

    # Oracle price = pool price from PREVIOUS BLOCK
    # Group by block, take last pool_price per block as that block's closing price
    block_prices = df.groupby('block_number')['pool_price'].last()
    df['prev_block'] = df['block_number'] - 1
    df['oracle_price'] = df['prev_block'].map(block_prices)

    # Forward-fill oracle price for blocks with no swap in previous block
    # Use the most recent available block price
    all_block_prices = block_prices.sort_index()
    df['oracle_price'] = df['block_number'].apply(
        lambda b: all_block_prices[all_block_prices.index < b].iloc[-1]
        if len(all_block_prices[all_block_prices.index < b]) > 0 else np.nan
    )

    df = df.dropna(subset=['oracle_price']).copy()

    # Determine trade direction and compute execution price
    if token0 in ('USDC', 'USDT'):
        # token0 is stablecoin
        # amount0 > 0 → stablecoin into pool → buying ETH
        # amount0 < 0 → stablecoin out of pool → selling ETH
        df['direction'] = np.where(df['amount0'] > 0, 'buy_eth', 'sell_eth')
        df['stablecoin_amount'] = df['amount0'].abs()
        df['eth_amount'] = df['amount1'].abs()
    else:
        # token0 is WETH
        # amount0 > 0 → WETH into pool → selling ETH
        # amount0 < 0 → WETH out of pool → buying ETH
        df['direction'] = np.where(df['amount0'] > 0, 'sell_eth', 'buy_eth')
        df['eth_amount'] = df['amount0'].abs()
        df['stablecoin_amount'] = df['amount1'].abs()

    # Filter out zero amounts
    df = df[(df['eth_amount'] > 0) & (df['stablecoin_amount'] > 0)].copy()

    # Execution price = stablecoin per ETH
    df['execution_price'] = df['stablecoin_amount'] / df['eth_amount']

    # Slippage vs oracle
    # buy_eth:  slippage = exec_price / oracle_price - 1  (positive = overpaid)
    # sell_eth: slippage = 1 - exec_price / oracle_price  (positive = undersold)
    df['slippage'] = np.where(
        df['direction'] == 'buy_eth',
        df['execution_price'] / df['oracle_price'] - 1,
        1 - df['execution_price'] / df['oracle_price'],
    )
    df['slippage_bps'] = df['slippage'] * 10000

    # Trade size in USD
    df['trade_size_usd'] = df['amountUSD'].abs()

    # Price impact (how much pool price moved)
    df['price_impact'] = (df['pool_price'] / df['oracle_price'] - 1).abs()
    df['price_impact_bps'] = df['price_impact'] * 10000

    # Pool metadata
    df['pool'] = pool_prefix
    df['fee_tier'] = fee_bps
    df['fee_pct'] = fee_bps / 10000
    df['pair'] = f"{token0}/{token1}"

    return df


def load_all_pools(base_dir: str) -> pd.DataFrame:
    """Load swap data from all pools."""
    all_dfs = []

    for pool_prefix in POOLS:
        data_dir = os.path.join(base_dir, f"swaps_2024_06_{pool_prefix}")
        if not os.path.exists(data_dir):
            print(f"  Skip {pool_prefix} (no data dir)")
            continue

        print(f"  Loading {pool_prefix} ({POOLS[pool_prefix][0]}/{POOLS[pool_prefix][1]} "
              f"{POOLS[pool_prefix][2]/10000:.2f}%)...")
        df = load_pool_swaps(data_dir, pool_prefix)
        if len(df) > 0:
            print(f"    → {len(df)} swaps, "
                  f"volume ${df['trade_size_usd'].sum():,.0f}")
            all_dfs.append(df)
        else:
            print(f"    → no valid swaps")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def analyze_slippage(df: pd.DataFrame):
    """Full slippage analysis."""

    # ── 1. Overall slippage by trade size ──
    bins = [0, 100, 1_000, 10_000, 50_000, 100_000, 500_000,
            1_000_000, 5_000_000, 10_000_000, float('inf')]
    labels = ['<$100', '$100-1k', '$1k-10k', '$10k-50k', '$50k-100k',
              '$100k-500k', '$500k-1M', '$1M-5M', '$5M-10M', '>$10M']

    df['size_bucket'] = pd.cut(df['trade_size_usd'], bins=bins, labels=labels)

    print(f"\n{'='*90}")
    print("SLIPPAGE BY TRADE SIZE (all pools combined)")
    print(f"{'='*90}\n")

    grouped = df.groupby('size_bucket', observed=True).agg(
        count=('slippage_bps', 'count'),
        mean_slip_bps=('slippage_bps', 'mean'),
        median_slip_bps=('slippage_bps', 'median'),
        p25_slip_bps=('slippage_bps', lambda x: x.quantile(0.25)),
        p75_slip_bps=('slippage_bps', lambda x: x.quantile(0.75)),
        p95_slip_bps=('slippage_bps', lambda x: x.quantile(0.95)),
        mean_impact_bps=('price_impact_bps', 'mean'),
        mean_size=('trade_size_usd', 'mean'),
    ).reset_index()

    print(grouped.to_string(index=False, float_format='{:.2f}'.format))

    # ── 2. Slippage by pool (fee tier) ──
    print(f"\n{'='*90}")
    print("SLIPPAGE BY POOL / FEE TIER")
    print(f"{'='*90}\n")

    pool_stats = df.groupby(['pool', 'pair', 'fee_pct'], observed=True).agg(
        count=('slippage_bps', 'count'),
        volume=('trade_size_usd', 'sum'),
        mean_slip_bps=('slippage_bps', 'mean'),
        median_slip_bps=('slippage_bps', 'median'),
        mean_size=('trade_size_usd', 'mean'),
    ).reset_index()

    print(pool_stats.to_string(index=False, float_format='{:.2f}'.format))

    # ── 3. Slippage by direction ──
    print(f"\n{'='*90}")
    print("SLIPPAGE BY DIRECTION")
    print(f"{'='*90}\n")

    dir_stats = df.groupby('direction', observed=True).agg(
        count=('slippage_bps', 'count'),
        mean_slip_bps=('slippage_bps', 'mean'),
        median_slip_bps=('slippage_bps', 'median'),
    ).reset_index()

    print(dir_stats.to_string(index=False, float_format='{:.2f}'.format))

    # ── 4. Specific trade sizes ──
    print(f"\n{'='*90}")
    print("EXACT SLIPPAGE FOR SPECIFIC TRADE SIZES")
    print(f"{'='*90}\n")

    print(f"{'Size':>12s}  {'N':>6s}  {'Median':>8s}  {'Mean':>8s}  "
          f"{'P5':>8s}  {'P95':>8s}  {'Impact':>8s}")
    print("-" * 70)

    for target in [10, 100, 1000, 5000, 10_000, 50_000, 100_000,
                   500_000, 1_000_000, 5_000_000, 10_000_000]:
        lo = target * 0.7
        hi = target * 1.3
        nearby = df[(df['trade_size_usd'] >= lo) & (df['trade_size_usd'] <= hi)]

        if len(nearby) >= 5:
            med = nearby['slippage_bps'].median()
            mean = nearby['slippage_bps'].mean()
            p5 = nearby['slippage_bps'].quantile(0.05)
            p95 = nearby['slippage_bps'].quantile(0.95)
            impact = nearby['price_impact_bps'].median()
            n = len(nearby)

            if target >= 1_000_000:
                label = f"${target/1_000_000:.0f}M"
            elif target >= 1000:
                label = f"${target/1000:.0f}k"
            else:
                label = f"${target}"

            print(f"{label:>12s}  {n:>6d}  {med:>7.2f}bp  {mean:>7.2f}bp  "
                  f"{p5:>7.2f}bp  {p95:>7.2f}bp  {impact:>7.2f}bp")
        else:
            label = f"${target:,}"
            print(f"{label:>12s}  {'<5 samples':>6s}")


    # ── 5. Factors that influence slippage ──
    print(f"\n{'='*90}")
    print("FACTORS INFLUENCING SLIPPAGE")
    print(f"{'='*90}\n")

    # a) Fee tier
    print("By fee tier:")
    for fee in sorted(df['fee_pct'].unique()):
        sub = df[df['fee_pct'] == fee]
        print(f"  {fee:.2f}% fee:  median={sub['slippage_bps'].median():.2f}bp  "
              f"mean={sub['slippage_bps'].mean():.2f}bp  n={len(sub)}")

    # b) Time of day (hour)
    df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
    print("\nBy hour of day (UTC):")
    hourly = df.groupby('hour').agg(
        median_slip=('slippage_bps', 'median'),
        mean_slip=('slippage_bps', 'mean'),
        count=('slippage_bps', 'count'),
        mean_size=('trade_size_usd', 'mean'),
    )
    # Show peak and trough hours
    peak = hourly['median_slip'].idxmax()
    trough = hourly['median_slip'].idxmin()
    print(f"  Worst hour:  {peak}:00 UTC  median={hourly.loc[peak, 'median_slip']:.2f}bp")
    print(f"  Best hour:   {trough}:00 UTC  median={hourly.loc[trough, 'median_slip']:.2f}bp")

    # c) Multiple swaps in same block (potential sandwiches)
    df['multi_swap_block'] = df.groupby('block_number')['block_number'].transform('count') > 1
    print("\nSingle vs multi-swap blocks:")
    for label, sub in df.groupby('multi_swap_block'):
        tag = "Multi-swap block" if label else "Single-swap block"
        print(f"  {tag:>22s}:  median={sub['slippage_bps'].median():.2f}bp  "
              f"mean={sub['slippage_bps'].mean():.2f}bp  n={len(sub)}")

    # d) Volatility proxy: price change in surrounding blocks
    df['price_vol'] = df['oracle_price'].pct_change().abs().rolling(10).mean() * 10000
    vol_bins = [0, 5, 20, 50, float('inf')]
    vol_labels = ['<5bp/block', '5-20bp/block', '20-50bp/block', '>50bp/block']
    df['vol_regime'] = pd.cut(df['price_vol'], bins=vol_bins, labels=vol_labels)
    print("\nBy volatility regime:")
    for regime, sub in df.groupby('vol_regime', observed=True):
        if len(sub) > 10:
            print(f"  {regime:>15s}:  median={sub['slippage_bps'].median():.2f}bp  "
                  f"mean={sub['slippage_bps'].mean():.2f}bp  n={len(sub)}")

    return df


def load_all_pools_for_month(base_dir: str, month_suffix: str, month_label: str) -> pd.DataFrame:
    """Load all pools for a given month suffix (e.g., '2024_06')."""
    all_dfs = []
    for pool_prefix in POOLS:
        data_dir = os.path.join(base_dir, f"swaps_{month_suffix}_{pool_prefix}")
        if not os.path.exists(data_dir):
            continue
        df = load_pool_swaps(data_dir, pool_prefix)
        if len(df) > 0:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined['month'] = month_label
        return combined
    return pd.DataFrame()


def compare_months(all_df: pd.DataFrame):
    """Compare slippage across months for the 0.05% fee pools."""
    # Focus on 0.05% fee pools (dominant volume)
    df05 = all_df[all_df['fee_pct'] == 0.05].copy()

    print(f"\n{'='*90}")
    print("SLIPPAGE COMPARISON ACROSS MONTHS (0.05% fee pools only)")
    print(f"{'='*90}\n")

    # Summary per month
    month_stats = df05.groupby('month', observed=True).agg(
        swaps=('slippage_bps', 'count'),
        volume=('trade_size_usd', 'sum'),
        median_slip=('slippage_bps', 'median'),
        mean_slip=('slippage_bps', 'mean'),
        p95_slip=('slippage_bps', lambda x: x.quantile(0.95)),
        mean_price=('oracle_price', 'mean'),
    ).reset_index()
    print(month_stats.to_string(index=False, float_format='{:.2f}'.format))

    # Size → slippage per month
    print(f"\n{'='*90}")
    print("MEDIAN SLIPPAGE (bps) BY SIZE AND MONTH (0.05% fee pools)")
    print(f"{'='*90}\n")

    targets = [1000, 10_000, 100_000, 500_000, 1_000_000]
    header = f"{'Size':>10s}"
    for m in sorted(df05['month'].unique()):
        header += f"  {m:>12s}"
    print(header)
    print("-" * len(header))

    for target in targets:
        lo, hi = target * 0.7, target * 1.3
        if target >= 1_000_000:
            label = f"${target/1_000_000:.0f}M"
        elif target >= 1000:
            label = f"${target/1000:.0f}k"
        else:
            label = f"${target}"

        row = f"{label:>10s}"
        for m in sorted(df05['month'].unique()):
            sub = df05[(df05['month'] == m) &
                       (df05['trade_size_usd'] >= lo) &
                       (df05['trade_size_usd'] <= hi)]
            if len(sub) >= 5:
                row += f"  {sub['slippage_bps'].median():>10.2f}bp"
            else:
                row += f"  {'n/a':>10s}  "
        print(row)

    # Curve slippage (subtract fee) per month
    print(f"\n{'='*90}")
    print("PURE CURVE SLIPPAGE (total - 5bp fee) BY SIZE AND MONTH")
    print(f"{'='*90}\n")

    print(header)
    print("-" * len(header))

    for target in targets:
        lo, hi = target * 0.7, target * 1.3
        if target >= 1_000_000:
            label = f"${target/1_000_000:.0f}M"
        elif target >= 1000:
            label = f"${target/1000:.0f}k"
        else:
            label = f"${target}"

        row = f"{label:>10s}"
        for m in sorted(df05['month'].unique()):
            sub = df05[(df05['month'] == m) &
                       (df05['trade_size_usd'] >= lo) &
                       (df05['trade_size_usd'] <= hi)]
            if len(sub) >= 5:
                curve = sub['slippage_bps'].median() - 5.0  # subtract fee
                row += f"  {curve:>10.2f}bp"
            else:
                row += f"  {'n/a':>10s}  "
        print(row)


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), 'data')

    months = [
        ('2023_10', 'Oct-2023'),
        ('2024_03', 'Mar-2024'),
        ('2024_06', 'Jun-2024'),
        ('2024_12', 'Dec-2024'),
    ]

    print("=" * 90)
    print("EMPIRICAL UNISWAP V3 SLIPPAGE ANALYSIS — MULTI-MONTH")
    print("Oracle: pool price from previous block")
    print("=" * 90)

    all_months = []
    for suffix, label in months:
        print(f"\n--- Loading {label} ---")
        df = load_all_pools_for_month(base_dir, suffix, label)
        if len(df) > 0:
            print(f"  Total: {len(df)} swaps, volume ${df['trade_size_usd'].sum():,.0f}")
            all_months.append(df)

    if not all_months:
        print("No data loaded!")
        exit(1)

    all_df = pd.concat(all_months, ignore_index=True)
    print(f"\n{'='*90}")
    print(f"GRAND TOTAL: {len(all_df)} swaps, {all_df['month'].nunique()} months, "
          f"volume ${all_df['trade_size_usd'].sum():,.0f}")
    print(f"{'='*90}")

    # Full analysis on combined data
    analyze_slippage(all_df)

    # Month-over-month comparison
    compare_months(all_df)
