"""
Fetch historical L2 book data from Hyperliquid S3 archive.
Processes tick-level snapshots into 6-hourly summaries.

Data available from 2023-04-15 onward.
Format: s3://hyperliquid-archive/market_data/[YYYYMMDD]/[hour]/l2Book/[COIN].lz4

Output schema:
    timestamp, venue, asset, market_type, depth_50bps, spread_bps, mid_price

Usage:
    python fetch_hl_l2_book.py --coin BTC --start 2024-01-01 --end 2025-12-28
    python fetch_hl_l2_book.py --coin ETH --start 2024-01-01 --end 2025-12-28
"""

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone


def download_and_decompress(date_str: str, hour: int, coin: str, tmp_dir: str) -> str | None:
    """Download and decompress a single hour of L2 book data. Returns path or None."""
    s3_path = f"s3://hyperliquid-archive/market_data/{date_str}/{hour}/l2Book/{coin}.lz4"
    lz4_path = os.path.join(tmp_dir, f"{date_str}_{hour}_{coin}.lz4")
    out_path = lz4_path.replace(".lz4", "")

    result = subprocess.run(
        ["aws", "s3", "cp", s3_path, lz4_path, "--request-payer", "requester"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None

    subprocess.run(["unlz4", "--rm", lz4_path], capture_output=True)
    if not os.path.exists(out_path):
        return None

    return out_path


def process_hour_file(filepath: str) -> dict | None:
    """Take one snapshot from the hour file and compute liquidity metrics."""
    with open(filepath, "r") as f:
        line = f.readline().strip()

    if not line:
        return None

    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    data = record["raw"]["data"]
    coin = data["coin"]
    time_ms = data["time"]
    levels = data["levels"]

    bids = levels[0]
    asks = levels[1]

    if not bids or not asks:
        return None

    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    mid_price = (best_bid + best_ask) / 2

    if mid_price == 0:
        return None

    spread_bps = (best_ask - best_bid) / mid_price * 10000

    threshold = mid_price * 0.005
    depth_bid = sum(
        float(lvl["sz"]) * float(lvl["px"])
        for lvl in bids
        if mid_price - float(lvl["px"]) <= threshold
    )
    depth_ask = sum(
        float(lvl["sz"]) * float(lvl["px"])
        for lvl in asks
        if float(lvl["px"]) - mid_price <= threshold
    )
    depth_50bps = depth_bid + depth_ask

    return {
        "timestamp": datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).isoformat(),
        "venue": "hyperliquid",
        "asset": coin,
        "market_type": "perp",
        "depth_50bps": round(depth_50bps, 2),
        "spread_bps": round(spread_bps, 4),
        "mid_price": mid_price,
    }


def fetch_one(date_str: str, hour: int, coin: str, tmp_dir: str) -> tuple[str, int, dict | None]:
    """Download, decompress, process one snapshot. Returns (date, hour, record)."""
    filepath = download_and_decompress(date_str, hour, coin, tmp_dir)
    if filepath is None:
        return (date_str, hour, None)
    record = process_hour_file(filepath)
    try:
        os.remove(filepath)
    except OSError:
        pass
    return (date_str, hour, record)


def date_range(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    while d < end_d:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Fetch HL L2 book from S3 archive")
    parser.add_argument("--coin", default="BTC", help="Coin (default: BTC)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-02-01", help="End date YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Output JSONL file path")
    parser.add_argument("--workers", type=int, default=16, help="Parallel downloads (default: 16)")
    args = parser.parse_args()

    if args.out:
        jsonl_path = args.out
    else:
        os.makedirs("liquidity_raw", exist_ok=True)
        jsonl_path = f"liquidity_raw/hl_{args.coin}_l2_book.jsonl"

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    # Resume support
    existing_dates = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing_dates.add(r["timestamp"][:10])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(existing_dates)} days already fetched")

    # Build task list
    tasks = []
    for date_str in date_range(args.start, args.end):
        iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if iso_date in existing_dates:
            continue
        for hour in range(0, 24, 6):
            tasks.append((date_str, hour))

    if not tasks:
        print("Nothing to fetch.")
        return

    print(f"Fetching {len(tasks)} snapshots with {args.workers} parallel workers...")

    total_count = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(jsonl_path, "a") as out_f:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(fetch_one, date_str, hour, args.coin, tmp_dir): (date_str, hour)
                    for date_str, hour in tasks
                }

                for future in as_completed(futures):
                    date_str, hour, record = future.result()
                    if record:
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        total_count += 1

                    if total_count % 100 == 0 and total_count > 0:
                        print(f"  {total_count} records written...")

    # Count total
    total_lines = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                total_lines += 1

    print(f"\nDone. {total_count} new + {len(existing_dates) * 4} existing = {total_lines} total records in {jsonl_path}")


if __name__ == "__main__":
    main()
