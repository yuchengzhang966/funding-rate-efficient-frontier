"""
Fetch Hyperliquid L2 book snapshots and compute simple liquidity metrics.

Emits rows compatible with:
    liquidity_data = pd.DataFrame({
        'timestamp': datetime,
        'venue': str,
        'asset': str,
        'market_type': str,     # 'spot' or 'perp'
        'depth_50bps': float,   # total bid+ask liquidity within 50bps of mid
        'spread_bps': float,    # bid-ask spread in basis points
        'mid_price': float
    })

Usage:
    python fetch_hl_liquidity.py --coin BTC --samples 5 --interval 2 --out liquidity.csv
    python fetch_hl_liquidity.py --coin ETH --secs 60 --interval 1 --format jsonl --out liquidity.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import requests


@dataclass
class BookSide:
    levels: list[dict]

    def prices(self) -> list[float]:
        out: list[float] = []
        for lvl in self.levels:
            try:
                out.append(float(lvl["px"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out


def _post_info(
    base_url: str,
    payload: dict[str, Any],
    timeout_s: int,
    max_retries: int,
    verify_tls: bool,
) -> Any:
    url = base_url.rstrip("/") + "/info"

    backoff_s = 0.5
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=timeout_s,
                headers={"Content-Type": "application/json"},
                verify=verify_tls,
            )
            if resp.status_code < 400:
                return resp.json()

            retryable = resp.status_code in (429, 500, 502, 503, 504)
            if (not retryable) or attempt > max_retries:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from /info after {attempt} attempts: {resp.text}"
                )
        except Exception:
            if attempt > max_retries:
                raise

        time.sleep(backoff_s)
        backoff_s = min(backoff_s * 2, 30.0)


def fetch_meta_sets(
    base_url: str, timeout_s: int, max_retries: int, verify_tls: bool
) -> tuple[set[str], set[str]]:
    meta = _post_info(
        base_url, {"type": "meta"}, timeout_s=timeout_s, max_retries=max_retries, verify_tls=verify_tls
    )
    spot = _post_info(
        base_url, {"type": "spotMeta"}, timeout_s=timeout_s, max_retries=max_retries, verify_tls=verify_tls
    )

    perp_set = set()
    spot_set = set()
    if isinstance(meta, dict):
        universe = meta.get("universe", [])
        if isinstance(universe, list):
            for row in universe:
                name = row.get("name") if isinstance(row, dict) else None
                if isinstance(name, str):
                    perp_set.add(name)

    if isinstance(spot, dict):
        universe = spot.get("universe", [])
        if isinstance(universe, list):
            for row in universe:
                name = row.get("name") if isinstance(row, dict) else None
                if isinstance(name, str):
                    spot_set.add(name)

    return perp_set, spot_set


def fetch_l2_book(
    base_url: str, coin: str, timeout_s: int, max_retries: int, verify_tls: bool
) -> dict[str, Any]:
    payload = {"type": "l2Book", "coin": coin}
    data = _post_info(
        base_url, payload, timeout_s=timeout_s, max_retries=max_retries, verify_tls=verify_tls
    )
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected l2Book response type: {type(data)}")
    return data


def _infer_sides(
    levels: list[list[dict]],
    side_hint: str | None,
) -> tuple[BookSide, BookSide]:
    if len(levels) != 2:
        raise RuntimeError(f"Expected 2 book sides, got {len(levels)}")
    side0 = BookSide(levels[0])
    side1 = BookSide(levels[1])

    if side_hint == "bids-first":
        return side0, side1
    if side_hint == "asks-first":
        return side1, side0

    max0 = max(side0.prices(), default=None)
    min0 = min(side0.prices(), default=None)
    max1 = max(side1.prices(), default=None)
    min1 = min(side1.prices(), default=None)

    if max0 is not None and min1 is not None and max0 <= min1:
        return side0, side1
    if max1 is not None and min0 is not None and max1 <= min0:
        return side1, side0

    p0 = side0.prices()[0] if side0.prices() else None
    p1 = side1.prices()[0] if side1.prices() else None
    if p0 is None or p1 is None:
        return side0, side1
    if p0 < p1:
        return side0, side1
    return side1, side0


def _best_prices(bids: BookSide, asks: BookSide) -> tuple[float | None, float | None]:
    bid_px = max(bids.prices(), default=None)
    ask_px = min(asks.prices(), default=None)
    return bid_px, ask_px


def _depth_within_bps(
    bids: BookSide,
    asks: BookSide,
    mid: float,
    depth_bps: float,
    depth_mode: str,
) -> float:
    if mid <= 0:
        return 0.0
    band = depth_bps / 10_000.0
    bid_floor = mid * (1.0 - band)
    ask_ceil = mid * (1.0 + band)

    def _accumulate(levels: Iterable[dict], price_check, mode: str) -> float:
        total = 0.0
        for lvl in levels:
            try:
                px = float(lvl["px"])
                sz = float(lvl["sz"])
            except (KeyError, TypeError, ValueError):
                continue
            if not price_check(px):
                continue
            total += sz * px if mode == "notional" else sz
        return total

    bid_total = _accumulate(bids.levels, lambda px: px >= bid_floor, depth_mode)
    ask_total = _accumulate(asks.levels, lambda px: px <= ask_ceil, depth_mode)
    return bid_total + ask_total


def _iso_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _coerce_ts_ms(ts_val: Any) -> int:
    if isinstance(ts_val, int):
        return ts_val
    if isinstance(ts_val, float):
        return int(ts_val)
    if isinstance(ts_val, str):
        try:
            return int(ts_val)
        except ValueError:
            return int(float(ts_val))
    raise RuntimeError(f"Unexpected timestamp type: {type(ts_val)}")


def _infer_market_type(
    coin: str,
    perp_set: set[str],
    spot_set: set[str],
    override: str | None,
) -> str:
    if override in ("spot", "perp"):
        return override
    in_perp = coin in perp_set
    in_spot = coin in spot_set
    if in_perp and not in_spot:
        return "perp"
    if in_spot and not in_perp:
        return "spot"
    if in_perp and in_spot:
        return "perp"
    return "unknown"


def _write_rows_csv(out_fp, rows: list[dict[str, Any]], write_header: bool) -> None:
    if not rows:
        return
    writer = csv.DictWriter(out_fp, fieldnames=list(rows[0].keys()))
    if write_header:
        writer.writeheader()
    for row in rows:
        writer.writerow(row)
    out_fp.flush()


def _write_rows_jsonl(out_fp, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        out_fp.write(json.dumps(row))
        out_fp.write("\n")
    out_fp.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid L2 liquidity metrics")
    parser.add_argument("--coin", default="BTC", help="Coin symbol (default: BTC)")
    parser.add_argument("--base-url", default="https://api.hyperliquid.xyz", help="API base URL.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to take.")
    parser.add_argument("--secs", type=int, default=0, help="Run for N seconds (overrides --samples).")
    parser.add_argument("--out", default="", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv", help="Output format.")
    parser.add_argument("--depth-bps", type=float, default=50.0, help="Depth band in bps.")
    parser.add_argument(
        "--depth-mode",
        choices=["notional", "size"],
        default="notional",
        help="Depth units: notional (px*size) or size (base units).",
    )
    parser.add_argument(
        "--side-hint",
        choices=["bids-first", "asks-first", "auto"],
        default="auto",
        help="Override book side ordering if needed.",
    )
    parser.add_argument(
        "--market-type",
        choices=["spot", "perp"],
        default=None,
        help="Override market_type if auto-detection is ambiguous.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout (seconds).")
    parser.add_argument("--max-retries", type=int, default=4, help="Max HTTP retries.")
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        default=True,
        help="Verify TLS certificates (default: true).",
    )
    parser.add_argument(
        "--no-verify-tls",
        action="store_false",
        dest="verify_tls",
        help="Disable TLS verification.",
    )
    args = parser.parse_args()

    perp_set, spot_set = fetch_meta_sets(
        args.base_url, timeout_s=args.timeout, max_retries=args.max_retries, verify_tls=args.verify_tls
    )
    market_type = _infer_market_type(args.coin, perp_set, spot_set, args.market_type)

    if market_type == "unknown":
        print(f"Warning: could not infer market_type for {args.coin}", file=sys.stderr)

    out_fp = sys.stdout
    close_fp = False
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_fp = open(args.out, "a", newline="")
        close_fp = True

    write_header = False
    if args.format == "csv":
        if args.out:
            try:
                write_header = os.path.getsize(args.out) == 0
            except OSError:
                write_header = True
        else:
            write_header = True

    deadline = time.time() + args.secs if args.secs and args.secs > 0 else None
    remaining = args.samples if not deadline else None
    side_hint = None if args.side_hint == "auto" else args.side_hint

    try:
        while True:
            if deadline and time.time() > deadline:
                break
            if remaining is not None and remaining <= 0:
                break

            book = fetch_l2_book(
                args.base_url, args.coin, timeout_s=args.timeout, max_retries=args.max_retries, verify_tls=args.verify_tls
            )
            levels = book.get("levels")
            ts_raw = book.get("time")
            if not isinstance(levels, list):
                raise RuntimeError(f"Unexpected l2Book payload: {book}")
            ts_ms = _coerce_ts_ms(ts_raw)

            bids, asks = _infer_sides(levels, side_hint)
            best_bid, best_ask = _best_prices(bids, asks)
            if best_bid is None or best_ask is None:
                raise RuntimeError("Could not determine best bid/ask from book levels.")

            mid = (best_bid + best_ask) / 2.0
            spread_bps = (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else 0.0
            depth = _depth_within_bps(
                bids, asks, mid=mid, depth_bps=args.depth_bps, depth_mode=args.depth_mode
            )

            row = {
                "timestamp": _iso_utc(ts_ms),
                "venue": "hyperliquid",
                "asset": args.coin,
                "market_type": market_type,
                "depth_50bps": depth,
                "spread_bps": spread_bps,
                "mid_price": mid,
            }

            if args.format == "csv":
                _write_rows_csv(out_fp, [row], write_header)
                write_header = False
            else:
                _write_rows_jsonl(out_fp, [row])

            if remaining is not None:
                remaining -= 1
            if args.interval > 0:
                time.sleep(args.interval)
    finally:
        if close_fp:
            out_fp.close()


if __name__ == "__main__":
    main()
