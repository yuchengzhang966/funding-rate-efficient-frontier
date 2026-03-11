"""
Fetch Binance perpetual funding rate history.
Binance funding rates are every 8 hours (3x/day).

Usage:
    python fetch_binance_funding.py --coin BTC --start 2022-01-01 --end 2025-12-30
    python fetch_binance_funding.py --coin ETH --start 2022-01-01 --end 2025-12-30
"""

import argparse
import json
import time
from datetime import datetime, timezone

import requests

BINANCE_FUNDING_URLS = [
    "https://www.binance.com/fapi/v1/fundingRate",   # Binance global (alt)
    "https://fapi.binance.com/fapi/v1/fundingRate",  # Binance global
]
LIMIT = 1000  # max per request


def _get_with_fallback(params: dict) -> list[dict]:
    for url in BINANCE_FUNDING_URLS:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 451:
                print(f"  Geo-blocked on {url}, trying next...")
                continue
            raise
    raise RuntimeError("All Binance endpoints returned 451. You may need a VPN.")


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    all_records = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        print(f"Fetching {symbol} from {cursor} ...")
        data = _get_with_fallback(params)

        if not data:
            break

        print(f"  Got {len(data)} records")
        all_records.extend(data)

        # move cursor past last record
        cursor = data[-1]["fundingTime"] + 1

        if len(data) < LIMIT:
            break

        time.sleep(0.2)

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance funding rates")
    parser.add_argument("--coin", default="BTC", help="Coin symbol (default: BTC)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-30", help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="funding_rate_raw", help="Output directory")
    args = parser.parse_args()

    symbol = f"{args.coin.upper()}USDT"
    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    records = fetch_funding_rates(symbol, start_ms, end_ms)

    # normalize to match your existing format
    output = [
        {
            "coin": args.coin.upper(),
            "funding_rate": r["fundingRate"],
            "time": r["fundingTime"],
        }
        for r in records
    ]

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"binance_{args.coin.upper()}_funding_rates.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(output)} records to {out_path}")


if __name__ == "__main__":
    main()