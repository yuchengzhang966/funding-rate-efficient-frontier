"""
Fetch borrow/supply rate data from Aave v3 API (Arbitrum).

No API key needed. Uses https://api.v3.aave.com/graphql
Fetches LAST_YEAR of daily data and duplicates it for 2022-2024 for simplicity.

Assets:
  - USDC, USDT: borrow rate
  - WETH, WBTC: supply rate (lending yield)

Usage:
    python fetch_defi_borrow_rates.py
    python fetch_defi_borrow_rates.py --asset USDC
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

AAVE_API = "https://api.v3.aave.com/graphql"

# Aave v3 Arbitrum pool + token addresses
MARKET = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
CHAIN_ID = 42161  # Arbitrum

ASSETS = {
    "USDC": {"address": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", "rate_type": "borrow"},
    "USDT": {"address": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9", "rate_type": "borrow"},
    "WETH": {"address": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1", "rate_type": "supply"},
    "WBTC": {"address": "0x2f2a2543B76A4166549F7aaB2e75Bef0aeFc5B0f", "rate_type": "supply"},
}


def fetch_apy_history(asset: str, rate_type: str) -> pd.DataFrame:
    """Fetch LAST_YEAR of daily APY data from Aave API."""
    token_addr = ASSETS[asset]["address"]

    if rate_type == "borrow":
        query_name = "borrowAPYHistory"
    else:
        query_name = "supplyAPYHistory"

    query = """
    {
      %s(request: {
        market: "%s",
        underlyingToken: "%s",
        window: LAST_YEAR,
        chainId: %d
      }) {
        date
        avgRate { value }
      }
    }
    """ % (query_name, MARKET, token_addr, CHAIN_ID)

    resp = requests.post(AAVE_API, json={"query": query}, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    if "errors" in result:
        raise RuntimeError(f"Aave API error: {result['errors']}")

    items = result["data"][query_name]
    rows = []
    for item in items:
        rows.append({
            "date": item["date"],
            "rate": float(item["avgRate"]["value"]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def replicate_for_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Duplicate the fetched year of data across multiple years."""
    # The fetched data covers ~last 365 days. We assign day-of-year indices
    # and replicate the rate pattern for each target year.
    df = df.copy()
    df["day_of_year"] = df["date"].dt.dayofyear

    all_frames = []
    for year in years:
        year_df = df.copy()
        # Map each row's day_of_year to the target year
        year_df["date"] = year_df["day_of_year"].apply(
            lambda d: datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=d - 1)
        )
        all_frames.append(year_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop(columns=["day_of_year"])
    return combined.sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch Aave v3 borrow/supply rates")
    parser.add_argument("--asset", default=None, help="Single asset: USDC, USDT, WETH, WBTC")
    parser.add_argument("--out-dir", default="borrow_rate_raw", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    assets_to_fetch = [args.asset] if args.asset else list(ASSETS.keys())
    target_years = [2022, 2023, 2024, 2025]

    all_rows = []

    for asset in assets_to_fetch:
        info = ASSETS[asset]
        rate_type = info["rate_type"]

        print(f"Fetching {asset} {rate_type} APY from Aave API ...")
        df = fetch_apy_history(asset, rate_type)
        print(f"  Got {len(df)} daily records")

        # Replicate across years
        df = replicate_for_years(df, target_years)
        df["asset"] = asset
        df["rate_type"] = rate_type
        df["protocol"] = "aave_arbitrum"

        all_rows.append(df)
        print(f"  -> {len(df)} records after replication for {target_years}")

    if not all_rows:
        print("No data fetched.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values(["asset", "date"]).reset_index(drop=True)

    # Save combined JSON
    out_path = os.path.join(args.out_dir, "defi_borrow_rates.json")
    combined.to_json(out_path, orient="records", date_format="iso", indent=2)
    print(f"\nWrote {len(combined)} total records to {out_path}")

    # Save per-asset CSVs
    for asset, group in combined.groupby("asset"):
        csv_path = os.path.join(args.out_dir, f"aave_arbitrum_{asset}_rates.csv")
        group.to_csv(csv_path, index=False)
        print(f"  {csv_path} ({len(group)} rows)")


if __name__ == "__main__":
    main()
