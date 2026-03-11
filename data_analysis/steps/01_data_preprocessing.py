"""
Data Preprocessing Pipeline
============================

This script loads and preprocesses all required data sources:
1. Hyperliquid funding rates (ETH, BTC)
2. Aave v3 Arbitrum USDC rates
3. Order book depth data (with mid_price for volatility calculation)
4. Volatility calculation from PRICE RETURNS (not funding rates!)

Input: Raw CSV files
Output: Clean master dataset ready for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Preprocess and merge all data sources."""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        
    def load_funding_rates(self, filepath):
        """
        Load Hyperliquid funding rate data.
        
        Expected CSV format:
            timestamp, asset, funding_rate
            2024-01-01 00:00:00, ETH, 0.00012
            2024-01-01 01:00:00, ETH, 0.00015
        
        Note: funding_rate should be the rate for that 1-hour period
        """
        print("Loading funding rate data...")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate
        assert 'timestamp' in df.columns, "Missing 'timestamp' column"
        assert 'asset' in df.columns, "Missing 'asset' column"
        assert 'funding_rate' in df.columns, "Missing 'funding_rate' column"
        
        # Sort
        df = df.sort_values(['asset', 'timestamp']).reset_index(drop=True)
        
        # Check for gaps
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset].copy()
            time_diffs = asset_df['timestamp'].diff()
            expected_diff = pd.Timedelta(hours=1)
            
            gaps = time_diffs[time_diffs > expected_diff * 1.5]
            if len(gaps) > 0:
                print(f"  Warning: {len(gaps)} gaps detected in {asset} data")
        
        # Annualize funding rates (1-hour rate → annual)
        # Annual rate = hourly_rate × 24 × 365
        df['funding_rate_annual'] = df['funding_rate'] * 24 * 365
        
        print(f"  Loaded {len(df)} funding rate records")
        print(f"  Assets: {df['asset'].unique()}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Mean funding rate (annual): {df['funding_rate_annual'].mean()*100:.2f}%")
        
        return df
    
    def load_aave_rates(self, filepath):
        """
        Load Aave v3 Arbitrum USDC rates.
        
        Expected CSV format:
            timestamp, supply_rate, borrow_rate, utilization, total_liquidity
            2024-01-01 00:00:00, 0.032, 0.048, 0.75, 800000000
        
        Rates should be annualized decimals (e.g., 0.048 = 4.8% APR)
        """
        print("\nLoading Aave rate data...")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate
        assert 'timestamp' in df.columns, "Missing 'timestamp' column"
        assert 'supply_rate' in df.columns, "Missing 'supply_rate' column"
        assert 'borrow_rate' in df.columns, "Missing 'borrow_rate' column"
        assert 'utilization' in df.columns, "Missing 'utilization' column"
        
        # Sort
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate net cost
        df['net_cost'] = df['borrow_rate'] - df['supply_rate']
        
        print(f"  Loaded {len(df)} rate snapshots")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Mean borrow rate: {df['borrow_rate'].mean()*100:.2f}%")
        print(f"  Mean supply rate: {df['supply_rate'].mean()*100:.2f}%")
        print(f"  Mean net cost: {df['net_cost'].mean()*100:.2f}%")
        print(f"  Mean utilization: {df['utilization'].mean()*100:.1f}%")
        
        return df
    
    def load_depth_data(self, filepath):
        """
        Load order book depth data.
        
        Expected CSV format:
            timestamp, asset, depth_spot, depth_perp, mid_price
            2024-01-01 00:00:00, ETH, 5000000, 5000000, 3500.0
        
        Depth = total liquidity within 50bps of mid price (USD)
        mid_price = required for volatility calculation
        """
        print("\nLoading depth data...")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate required columns
        required = ['timestamp', 'asset', 'depth_spot', 'depth_perp', 'mid_price']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.sort_values(['asset', 'timestamp']).reset_index(drop=True)
        
        print(f"  Loaded {len(df)} depth snapshots")
        print(f"  Mean spot depth: ${df['depth_spot'].mean():,.0f}")
        print(f"  Mean perp depth: ${df['depth_perp'].mean():,.0f}")
        print(f"  Mean mid price: ${df['mid_price'].mean():,.2f}")
        
        # Check for NaN in critical columns
        for col in ['depth_spot', 'depth_perp', 'mid_price']:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️  Warning: {nan_count} NaN values in {col}")
        
        return df
    
    def calculate_volatility(self, depth_df, window_hours=24):
        """
        Calculate realized volatility from price returns.
        
        CRITICAL: Must use actual price data, not funding rates!
        Uses mid_price from depth data.
        
        Args:
            depth_df: DataFrame with mid_price column
            window_hours: Rolling window for volatility calculation
        """
        print(f"\nCalculating {window_hours}h realized volatility from price returns...")
        
        vol_data = []
        
        for asset in depth_df['asset'].unique():
            asset_df = depth_df[depth_df['asset'] == asset].copy()
            
            # CRITICAL: Calculate log returns from PRICES, not funding rates
            asset_df['log_return'] = np.log(asset_df['mid_price'] / asset_df['mid_price'].shift(1))
            
            # Infer observation frequency from median time delta
            dt = asset_df['timestamp'].diff().dt.total_seconds().median()
            obs_per_year = 365.25 * 24 * 3600 / dt if dt and dt > 0 else 8760

            # Rolling volatility (annualized)
            # std of per-observation returns × sqrt(obs_per_year)
            asset_df['realized_vol'] = (
                asset_df['log_return']
                .rolling(window=window_hours)
                .std() * np.sqrt(obs_per_year)
            )
            
            vol_data.append(asset_df[['timestamp', 'asset', 'realized_vol']])
        
        vol_df = pd.concat(vol_data, ignore_index=True)
        
        # Fill initial NaN values with median
        for asset in vol_df['asset'].unique():
            mask = vol_df['asset'] == asset
            median_vol = vol_df.loc[mask, 'realized_vol'].median()
            vol_df.loc[mask, 'realized_vol'] = vol_df.loc[mask, 'realized_vol'].fillna(median_vol)
        
        print(f"  Calculated volatility for {vol_df['asset'].nunique()} assets")
        print(f"  Mean volatility: {vol_df['realized_vol'].mean()*100:.1f}%")
        
        return vol_df
    
    def merge_all_data(self, funding_df, aave_dfs, depth_df, vol_df):
        """
        Merge all data sources into master dataset.

        CRITICAL: Handles mismatched frequencies properly:
        - Funding: hourly
        - Depth/vol: 6-hourly
        - Aave: daily

        Uses merge_asof with forward-fill to avoid NaN propagation.

        Args:
            funding_df: Funding rate DataFrame
            aave_dfs: Either a single DataFrame (applied to all assets) or a
                      dict mapping asset name -> DataFrame for per-asset rates
            depth_df: Order book depth DataFrame
            vol_df: Realized volatility DataFrame

        Result: One row per (timestamp, asset) with all features.
        """
        print("\nMerging all data sources...")

        # Normalize aave_dfs to a dict
        if isinstance(aave_dfs, pd.DataFrame):
            aave_dict = {asset: aave_dfs for asset in funding_df['asset'].unique()}
        else:
            aave_dict = aave_dfs

        # Start with funding rates (hourly - most granular for this asset)
        master = funding_df.copy()

        # Add volatility (merge_asof to handle 6-hour vs 1-hour)
        master = master.sort_values(['asset', 'timestamp'])
        vol_df = vol_df.sort_values(['asset', 'timestamp'])

        # Merge within each asset group
        merged_list = []
        for asset in master['asset'].unique():
            asset_funding = master[master['asset'] == asset].reset_index(drop=True)
            asset_vol = vol_df[vol_df['asset'] == asset].reset_index(drop=True)

            # merge_asof: for each funding timestamp, use most recent vol
            merged_asset = pd.merge_asof(
                asset_funding,
                asset_vol[['timestamp', 'realized_vol']],
                on='timestamp',
                direction='backward'
            )
            merged_list.append(merged_asset)

        master = pd.concat(merged_list, ignore_index=True)

        # Add depth (same approach - handle 6-hour vs 1-hour)
        depth_df = depth_df.sort_values(['asset', 'timestamp'])

        merged_list = []
        for asset in master['asset'].unique():
            asset_master = master[master['asset'] == asset].reset_index(drop=True)
            asset_depth = depth_df[depth_df['asset'] == asset].reset_index(drop=True)

            # merge_asof: for each funding timestamp, use most recent depth
            merged_asset = pd.merge_asof(
                asset_master,
                asset_depth[['timestamp', 'depth_spot', 'depth_perp']],
                on='timestamp',
                direction='backward'
            )
            merged_list.append(merged_asset)

        master = pd.concat(merged_list, ignore_index=True)

        # Add Aave rates per asset (merge_asof for nearest timestamp)
        merged_list = []
        for asset in master['asset'].unique():
            asset_master = master[master['asset'] == asset].copy()
            asset_master = asset_master.sort_values('timestamp').reset_index(drop=True)

            aave_df = aave_dict.get(asset)
            if aave_df is not None:
                aave_df = aave_df.sort_values('timestamp')
                asset_master = pd.merge_asof(
                    asset_master,
                    aave_df,
                    on='timestamp',
                    direction='backward'
                )
            merged_list.append(asset_master)

        master = pd.concat(merged_list, ignore_index=True)

        print(f"  Master dataset: {len(master)} rows")
        print(f"  Columns: {list(master.columns)}")

        # Check for missing values AFTER merge
        critical_cols = ['funding_rate_annual', 'realized_vol', 'depth_spot',
                        'depth_perp', 'borrow_rate', 'supply_rate']

        print("\n  Data completeness:")
        for col in critical_cols:
            if col in master.columns:
                nan_count = master[col].isna().sum()
                nan_pct = (nan_count / len(master)) * 100
                print(f"    {col}: {nan_count} NaN ({nan_pct:.1f}%)")

                if nan_pct > 5:
                    print(f"      ⚠️  WARNING: >5% missing data in {col}")

        # Forward-fill any remaining NaN in depth/vol (from start of series)
        master['realized_vol'] = master.groupby('asset')['realized_vol'].bfill()
        master['depth_spot'] = master.groupby('asset')['depth_spot'].bfill()
        master['depth_perp'] = master.groupby('asset')['depth_perp'].bfill()

        return master
    
    def add_regime_labels(self, master_df):
        """
        Add regime labels for stratified analysis.
        
        Regimes:
        - Volatility: Low (<40%), Medium (40-60%), High (>60%)
        - Liquidity: Deep (>median), Thin (<median)
        - Funding: Positive (>0), Negative (<0)
        """
        print("\nAdding regime labels...")
        
        df = master_df.copy()
        
        # Volatility regime
        df['vol_regime'] = pd.cut(
            df['realized_vol'],
            bins=[0, 0.40, 0.60, np.inf],
            labels=['low_vol', 'med_vol', 'high_vol']
        )
        
        # Liquidity regime (use combined spot + perp depth)
        df['total_depth'] = df['depth_spot'] + df['depth_perp']
        median_depth = df['total_depth'].median()
        df['liq_regime'] = df['total_depth'].apply(
            lambda x: 'deep' if x >= median_depth else 'thin'
        )
        
        # Funding regime
        df['funding_regime'] = df['funding_rate'].apply(
            lambda x: 'positive' if x >= 0 else 'negative'
        )
        
        # Combined regime
        df['regime'] = (
            df['vol_regime'].astype(str) + '_' +
            df['liq_regime'].astype(str) + '_' +
            df['funding_regime'].astype(str)
        )
        
        print(f"  Created {df['regime'].nunique()} unique regimes")
        print("\n  Regime distribution:")
        print(df['regime'].value_counts().head(10))
        
        return df
    
    def run_full_pipeline(self, funding_file, aave_files, depth_file, output_file):
        """
        Run complete preprocessing pipeline.

        Args:
            funding_file: Path to Hyperliquid funding rate CSV
            aave_files: Either a single path (str) or a dict mapping
                        asset name -> path for per-asset Aave rate CSVs
            depth_file: Path to depth data CSV (MUST have mid_price column)
            output_file: Path to save processed master dataset
        """
        print("="*70)
        print("FUNDING VAULT DATA PREPROCESSING PIPELINE")
        print("="*70)

        # Load data
        funding_df = self.load_funding_rates(funding_file)

        # Load Aave rates (single file or per-asset)
        if isinstance(aave_files, str):
            aave_dfs = self.load_aave_rates(aave_files)
        else:
            aave_dfs = {}
            for asset, path in aave_files.items():
                print(f"\nLoading Aave rates for {asset}...")
                aave_dfs[asset] = self.load_aave_rates(path)

        depth_df = self.load_depth_data(depth_file)

        # CRITICAL: Calculate volatility from PRICES in depth data, not funding rates
        vol_df = self.calculate_volatility(depth_df)

        # Merge
        master_df = self.merge_all_data(funding_df, aave_dfs, depth_df, vol_df)

        # Add regimes
        master_df = self.add_regime_labels(master_df)

        # Save
        master_df.to_csv(output_file, index=False)
        print(f"\nSaved master dataset to: {output_file}")
        print(f"   Shape: {master_df.shape}")

        # Summary statistics
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"\nDate range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
        print(f"Duration: {(master_df['timestamp'].max() - master_df['timestamp'].min()).days} days")
        print(f"Assets: {master_df['asset'].unique()}")
        print(f"Total observations: {len(master_df):,}")

        print("\nFunding rates (annual):")
        print(master_df.groupby('asset')['funding_rate_annual'].describe())

        print("\nVolatility (annual):")
        print(master_df.groupby('asset')['realized_vol'].describe())

        print("\nAave rates per asset:")
        print(master_df.groupby('asset')[['borrow_rate', 'supply_rate']].mean())

        return master_df


def create_sample_data():
    """
    Create sample datasets for testing.
    
    Use this if you don't have real data yet.
    """
    print("Creating sample data for testing...")
    
    # Date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 28)
    dates = pd.date_range(start_date, end_date, freq='1H')
    
    # Sample funding rates
    funding_data = []
    for asset in ['ETH', 'BTC']:
        for date in dates:
            # Simulate funding rate (mean-reverting random walk)
            funding_rate = np.random.normal(0.00001, 0.00003)  # 1h rate
            funding_data.append({
                'timestamp': date,
                'asset': asset,
                'funding_rate': funding_rate
            })
    
    funding_df = pd.DataFrame(funding_data)
    funding_df.to_csv('./data/sample_funding_rates.csv', index=False)
    
    # Sample Aave rates
    aave_data = []
    for date in dates[::24]:  # Daily snapshots
        util = np.random.uniform(0.70, 0.85)
        borrow_rate = 0.04 + 0.01 * util if util < 0.80 else 0.04 + 0.60 * (util - 0.80)
        supply_rate = util * borrow_rate * 0.90
        
        aave_data.append({
            'timestamp': date,
            'supply_rate': supply_rate,
            'borrow_rate': borrow_rate,
            'utilization': util,
            'total_liquidity': 800_000_000
        })
    
    aave_df = pd.DataFrame(aave_data)
    aave_df.to_csv('./data/sample_aave_rates.csv', index=False)
    
    # Sample depth data
    depth_data = []
    for asset in ['ETH', 'BTC']:
        for date in dates[::6]:  # Every 6 hours
            depth_data.append({
                'timestamp': date,
                'asset': asset,
                'depth_spot': np.random.uniform(4_000_000, 6_000_000),
                'depth_perp': np.random.uniform(4_000_000, 6_000_000)
            })
    
    depth_df = pd.DataFrame(depth_data)
    depth_df.to_csv('./data/sample_depth_data.csv', index=False)
    
    print("✅ Sample data created in ./data/")


if __name__ == '__main__':
    """
    Usage:
    
    1. If you have real data:
       - Place CSVs in ./data/ directory
       - Update file paths below
       - Run: python 01_data_preprocessing.py
    
    2. If you want to test with sample data:
       - Uncomment the create_sample_data() line
       - Run: python 01_data_preprocessing.py
    """
    
    import os
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Option 1: Create sample data for testing
    # create_sample_data()
    
    # Option 2: Process real data
    preprocessor = DataPreprocessor(data_dir='./data')

    # Per-asset Aave files: ETH uses WETH supply, BTC uses WBTC supply
    aave_files = {
        'ETH': './data/aave_rates_eth.csv',
        'BTC': './data/aave_rates_btc.csv',
    }

    # Fall back to single file if per-asset files don't exist
    if not os.path.exists(aave_files['ETH']):
        aave_files = './data/aave_v3_arbitrum_usdc_rates.csv'

    master_df = preprocessor.run_full_pipeline(
        funding_file='./data/hyperliquid_funding_rates.csv',
        aave_files=aave_files,
        depth_file='./data/hyperliquid_depth_data.csv',
        output_file='./data/master_dataset.csv'
    )

    print("\nPreprocessing complete!")