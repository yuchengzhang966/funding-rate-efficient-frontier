"""
Data Verification Script
=========================

Run this BEFORE running the pipeline to check your data is correctly formatted.

Usage:
    python verify_data.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def verify_depth_data(filepath='./data/hyperliquid_depth_data.csv'):
    """Verify depth data format and quality."""
    
    print("\n" + "="*70)
    print("VERIFYING DEPTH DATA")
    print("="*70)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n✅ Loaded {len(df)} rows from {filepath}")
    
    # Check 1: Required columns
    required = ['timestamp', 'asset', 'depth_spot', 'depth_perp', 'mid_price']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\n❌ CRITICAL: Missing required columns: {missing}")
        print(f"   Found columns: {list(df.columns)}")
        print(f"\n   Your CSV must have these exact columns:")
        print(f"   timestamp, asset, depth_spot, depth_perp, mid_price")
        return False
    else:
        print(f"\n✅ All required columns present")
    
    # Check 2: Parse timestamp
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Timestamps parsed successfully")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except:
        print(f"❌ CRITICAL: Could not parse timestamps")
        return False
    
    # Check 3: NaN values
    print(f"\n Checking for NaN values...")
    has_nan = False
    for col in ['depth_spot', 'depth_perp', 'mid_price']:
        nan_count = df[col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        
        if nan_count > 0:
            print(f"   ⚠️  {col}: {nan_count} NaN values ({nan_pct:.1f}%)")
            has_nan = True
        else:
            print(f"   ✅ {col}: No NaN values")
    
    if has_nan:
        print(f"\n   ⚠️  WARNING: NaN values detected. Will be forward-filled during preprocessing.")
    
    # Check 4: Depth values reasonable
    print(f"\n Checking depth values...")
    for col in ['depth_spot', 'depth_perp']:
        mean_depth = df[col].mean()
        min_depth = df[col].min()
        max_depth = df[col].max()
        
        if mean_depth < 100000:
            print(f"   ⚠️  {col} mean = ${mean_depth:,.0f} - seems low (expected >$100k)")
        elif mean_depth > 100_000_000:
            print(f"   ⚠️  {col} mean = ${mean_depth:,.0f} - seems high (expected <$100M)")
        else:
            print(f"   ✅ {col} mean = ${mean_depth:,.0f} (reasonable)")
    
    # Check 5: Calculate and verify volatility
    print(f"\n Verifying price volatility calculation...")
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset].sort_values('timestamp')
        
        if len(asset_df) < 24:
            print(f"   ⚠️  {asset}: Only {len(asset_df)} observations (need at least 24)")
            continue
        
        # Calculate log returns
        asset_df['log_return'] = np.log(asset_df['mid_price'] / asset_df['mid_price'].shift(1))
        
        # Annualized volatility
        vol_annual = asset_df['log_return'].std() * np.sqrt(8760)
        
        if vol_annual < 0.20:
            print(f"   ⚠️  {asset} volatility: {vol_annual*100:.1f}% - unusually low for crypto")
        elif vol_annual > 1.50:
            print(f"   ⚠️  {asset} volatility: {vol_annual*100:.1f}% - unusually high")
        else:
            print(f"   ✅ {asset} volatility: {vol_annual*100:.1f}% (typical crypto range: 40-80%)")
    
    # Check 6: Sampling frequency
    print(f"\n Checking sampling frequency...")
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset].sort_values('timestamp')
        time_diffs = asset_df['timestamp'].diff().dt.total_seconds() / 3600  # hours
        median_interval = time_diffs.median()
        
        print(f"   {asset} median interval: {median_interval:.1f} hours")
        
        if median_interval < 0.5:
            print(f"      ⚠️  Very high frequency (sub-hourly)")
        elif median_interval > 24:
            print(f"      ⚠️  Low frequency (>24h) - may want denser sampling")
    
    print(f"\n" + "="*70)
    print(f"DEPTH DATA VERIFICATION: {'✅ PASSED' if not missing else '❌ FAILED'}")
    print(f"="*70)
    
    return len(missing) == 0


def verify_funding_data(filepath='./data/hyperliquid_funding_rates.csv'):
    """Verify funding rate data."""
    
    print("\n" + "="*70)
    print("VERIFYING FUNDING RATE DATA")
    print("="*70)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    df = pd.read_csv(filepath)
    print(f"\n✅ Loaded {len(df)} rows")
    
    # Check columns
    required = ['timestamp', 'asset', 'funding_rate']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    else:
        print(f"✅ All required columns present")
    
    # Check funding rate range
    df['funding_rate_annual'] = df['funding_rate'] * 24 * 365
    
    print(f"\n Funding rate statistics (annualized):")
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset]
        mean_rate = asset_df['funding_rate_annual'].mean()
        min_rate = asset_df['funding_rate_annual'].min()
        max_rate = asset_df['funding_rate_annual'].max()
        
        print(f"   {asset}: mean={mean_rate*100:.2f}%, range=[{min_rate*100:.2f}%, {max_rate*100:.2f}%]")
        
        if abs(mean_rate) > 0.50:
            print(f"      ⚠️  Mean funding rate seems extreme")
    
    print(f"\n✅ FUNDING DATA VERIFICATION PASSED")
    
    return True


def verify_aave_data(filepath=None):
    """Verify Aave rate data. Supports per-asset files or single file."""

    print("\n" + "="*70)
    print("VERIFYING AAVE RATE DATA")
    print("="*70)

    # Detect per-asset or single file
    if filepath is None:
        per_asset = {
            'ETH': './data/aave_rates_eth.csv',
            'BTC': './data/aave_rates_btc.csv',
        }
        single = './data/aave_v3_arbitrum_usdc_rates.csv'

        if os.path.exists(per_asset['ETH']):
            filepaths = per_asset
        elif os.path.exists(single):
            filepaths = {'ALL': single}
        else:
            print(f"  File not found: tried {list(per_asset.values())} and {single}")
            return False
    else:
        filepaths = {'ALL': filepath}

    all_ok = True
    for label, fpath in filepaths.items():
        if not os.path.exists(fpath):
            print(f"  File not found: {fpath}")
            all_ok = False
            continue

        df = pd.read_csv(fpath)
        print(f"\n  [{label}] Loaded {len(df)} rows from {fpath}")

        required = ['timestamp', 'supply_rate', 'borrow_rate', 'utilization']
        missing = [col for col in required if col not in df.columns]

        if missing:
            print(f"  Missing columns: {missing}")
            all_ok = False
            continue

        print(f"  All required columns present")
        print(f"  Borrow rate: {df['borrow_rate'].mean()*100:.2f}% (mean)")
        print(f"  Supply rate: {df['supply_rate'].mean()*100:.2f}% (mean)")
        print(f"  Utilization: {df['utilization'].mean()*100:.1f}% (mean)")

    status = "PASSED" if all_ok else "FAILED"
    print(f"\n  AAVE DATA VERIFICATION: {status}")

    return all_ok


def main():
    """Run all verifications."""
    
    print("\n" + "="*70)
    print("DATA VERIFICATION SUITE")
    print("="*70)
    print("\nThis script checks your data files are correctly formatted")
    print("BEFORE you run the main pipeline.\n")
    
    results = {}
    
    # Check each data source
    results['depth'] = verify_depth_data()
    results['funding'] = verify_funding_data()
    results['aave'] = verify_aave_data()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL CHECKS PASSED - READY TO RUN PIPELINE!")
        print("="*70)
        print("\nNext step:")
        print("  python run_pipeline_simple.py")
    else:
        print("\n" + "="*70)
        print("❌ SOME CHECKS FAILED - FIX DATA BEFORE RUNNING")
        print("="*70)
        print("\nFix the issues above, then run this script again.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    import sys
    sys.exit(0 if success else 1)