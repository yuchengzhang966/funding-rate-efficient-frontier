"""
Transform raw fetched data into the CSV format expected by 01_data_preprocessing.py.

Raw sources:
    funding_rate_raw/hl_BTC_funding_rates.json
    funding_rate_raw/hl_ETH_funding_rates.json
    borrow_rate_raw/aave_arbitrum_USDC_rates.csv
    borrow_rate_raw/aave_arbitrum_USDT_rates.csv
    borrow_rate_raw/aave_arbitrum_WETH_rates.csv
    liquidity_raw/hl_BTC_l2_book.jsonl
    liquidity_raw/hl_ETH_l2_book.jsonl

Output (into steps/data/):
    hyperliquid_funding_rates.csv   — columns: timestamp, asset, funding_rate
    aave_v3_arbitrum_usdc_rates.csv — columns: timestamp, supply_rate, borrow_rate, utilization, total_liquidity
    hyperliquid_depth_data.csv      — columns: timestamp, asset, depth_spot, depth_perp, mid_price

Usage:
    cd data_analysis
    python transform_raw_data.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd


RAW_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(RAW_DIR, "steps", "data")


def transform_funding_rates():
    """Merge HL BTC + ETH funding data into one CSV.

    Supports both JSON arrays (funding_rate_raw/*.json) and JSONL
    (funding_out/*.jsonl) — whichever is found first.
    """
    print("=== Funding Rates ===")

    candidates = [
        # JSONL in funding_out/
        (os.path.join(RAW_DIR, "..", "funding_out", "funding_rates_BTC.jsonl"), "jsonl"),
        (os.path.join(RAW_DIR, "..", "funding_out", "funding_rates_ETH.jsonl"), "jsonl"),
        # JSON arrays in funding_rate_raw/
        (os.path.join(RAW_DIR, "funding_rate_raw", "hl_BTC_funding_rates.json"), "json"),
        (os.path.join(RAW_DIR, "funding_rate_raw", "hl_ETH_funding_rates.json"), "json"),
    ]

    files = []
    for path, fmt in candidates:
        if os.path.exists(path):
            files.append((path, fmt))
    if not files:
        raise FileNotFoundError("No funding rate files found in funding_out/ or funding_rate_raw/")

    rows = []
    for fpath, fmt in files:
        if fmt == "jsonl":
            data = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(fpath) as f:
                data = json.load(f)
        print(f"  Loaded {len(data)} records from {os.path.basename(fpath)}")
        for r in data:
            rows.append({
                "timestamp": datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc)
                    .strftime("%Y-%m-%d %H:%M:%S"),
                "asset": r["coin"],
                "funding_rate": float(r["funding_rate"]),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["asset", "timestamp"])

    out = os.path.join(OUT_DIR, "hyperliquid_funding_rates.csv")
    df.to_csv(out, index=False)
    print(f"  -> {out}  ({len(df)} rows, assets: {sorted(df['asset'].unique())})")
    return df


def transform_aave_rates():
    """
    Produce per-asset Aave rate files.

    borrow_rate = min(USDC, USDT) borrow rate on each date (shared)
    supply_rate = WETH for ETH, WBTC for BTC

    Output:
        aave_rates_eth.csv  — borrow_rate + WETH supply_rate
        aave_rates_btc.csv  — borrow_rate + WBTC supply_rate
    """
    print("\n=== Aave Rates ===")

    borrow_dir = os.path.join(RAW_DIR, "borrow_rate_raw")

    # Load borrow rates (USDC + USDT) — shared across assets
    usdc = pd.read_csv(os.path.join(borrow_dir, "aave_arbitrum_USDC_rates.csv"))
    usdt = pd.read_csv(os.path.join(borrow_dir, "aave_arbitrum_USDT_rates.csv"))
    print(f"  USDC borrow: {len(usdc)} rows, USDT borrow: {len(usdt)} rows")

    usdc["date"] = pd.to_datetime(usdc["date"], utc=True)
    usdt["date"] = pd.to_datetime(usdt["date"], utc=True)

    borrow = pd.merge(
        usdc[["date", "rate"]].rename(columns={"rate": "usdc_borrow"}),
        usdt[["date", "rate"]].rename(columns={"rate": "usdt_borrow"}),
        on="date",
        how="outer",
    )
    borrow["borrow_rate"] = borrow[["usdc_borrow", "usdt_borrow"]].min(axis=1)

    # Load supply rates
    supply_files = {
        "ETH": "aave_arbitrum_WETH_rates.csv",
        "BTC": "aave_arbitrum_WBTC_rates.csv",
    }

    for asset, fname in supply_files.items():
        supply = pd.read_csv(os.path.join(borrow_dir, fname))
        print(f"  {asset} supply ({fname}): {len(supply)} rows")
        supply["date"] = pd.to_datetime(supply["date"], utc=True)

        merged = pd.merge(
            borrow[["date", "borrow_rate"]],
            supply[["date", "rate"]].rename(columns={"rate": "supply_rate"}),
            on="date",
            how="outer",
        )
        merged = merged.sort_values("date").reset_index(drop=True)
        merged["borrow_rate"] = merged["borrow_rate"].ffill().bfill()
        merged["supply_rate"] = merged["supply_rate"].ffill().bfill()

        # Defaults for columns the pipeline requires but Aave API doesn't provide
        merged["utilization"] = 0.80
        merged["total_liquidity"] = 800_000_000

        merged["timestamp"] = merged["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out_df = merged[["timestamp", "supply_rate", "borrow_rate", "utilization", "total_liquidity"]]

        out = os.path.join(OUT_DIR, f"aave_rates_{asset.lower()}.csv")
        out_df.to_csv(out, index=False)
        print(f"  -> {out}  ({len(out_df)} rows)")
        print(f"     Mean borrow: {out_df['borrow_rate'].mean()*100:.2f}%  Mean supply: {out_df['supply_rate'].mean()*100:.2f}%")


def transform_depth_data():
    """
    Convert L2 book JSONL into depth CSV.

    depth_spot = depth_perp (no spot orderbook from HL, use perp as proxy)
    depth_perp = depth_50bps from raw data
    """
    print("\n=== Depth Data ===")

    files = [
        os.path.join(RAW_DIR, "liquidity_raw", "hl_BTC_l2_book.jsonl"),
        os.path.join(RAW_DIR, "liquidity_raw", "hl_ETH_l2_book.jsonl"),
    ]

    rows = []
    for fpath in files:
        count = 0
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ts = pd.Timestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                rows.append({
                    "timestamp": ts,
                    "asset": r["asset"],
                    "depth_spot": r["depth_50bps"],   # use perp depth as spot proxy
                    "depth_perp": r["depth_50bps"],
                    "mid_price": r["mid_price"],
                })
                count += 1
        print(f"  Loaded {count} records from {os.path.basename(fpath)}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["asset", "timestamp"])

    out = os.path.join(OUT_DIR, "hyperliquid_depth_data.csv")
    df.to_csv(out, index=False)
    print(f"  -> {out}  ({len(df)} rows, assets: {sorted(df['asset'].unique())})")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Output directory: {OUT_DIR}\n")

    transform_funding_rates()
    transform_aave_rates()
    transform_depth_data()

    print("\nDone. Files ready for 01_data_preprocessing.py")


if __name__ == "__main__":
    main()