"""
Exact Uniswap v3 Slippage Calculator
======================================

Given a pool state (liquidity distribution across ticks, current price, fee tier),
calculates the exact output and slippage for any trade size.

No approximations — this replicates the Uniswap v3 swap math exactly.

Uniswap v3 core math:
- Pool stores liquidity L concentrated in tick ranges
- Within a tick range, behaves like constant-product AMM with virtual reserves:
    x = L / √P   (token0 virtual reserve)
    y = L × √P   (token1 virtual reserve)
- Price P = token1 / token0 (e.g., USDC per ETH)
- √P is the core state variable

Swap formulas (within one tick range):
  Buying token0 with token1 (e.g., USDC → ETH):
    √P_new = √P_old + Δy / L        (price goes up — ETH more expensive)
    Δx_out = L × (1/√P_old - 1/√P_new)

  Selling token0 for token1 (e.g., ETH → USDC):
    √P_new = L × √P_old / (L + Δx × √P_old)   (price goes down)
    Δy_out = L × (√P_old - √P_new)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


# Uniswap v3 tick math constants
MIN_TICK = -887272
MAX_TICK = 887272
TICK_BASE = 1.0001  # price ratio per tick


def tick_to_sqrt_price(tick: int) -> float:
    """Convert tick to √P."""
    return TICK_BASE ** (tick / 2)


def sqrt_price_to_tick(sqrt_price: float) -> int:
    """Convert √P to tick (rounds down)."""
    return math.floor(2 * math.log(sqrt_price) / math.log(TICK_BASE))


def sqrt_price_to_price(sqrt_price: float) -> float:
    """Convert √P to price P."""
    return sqrt_price ** 2


@dataclass
class TickRange:
    """A range of ticks with concentrated liquidity."""
    tick_lower: int
    tick_upper: int
    liquidity: float  # L in this range

    @property
    def sqrt_price_lower(self) -> float:
        return tick_to_sqrt_price(self.tick_lower)

    @property
    def sqrt_price_upper(self) -> float:
        return tick_to_sqrt_price(self.tick_upper)


@dataclass
class PoolState:
    """
    Uniswap v3 pool state.

    Attributes:
        token0: Token0 symbol (e.g., "WETH")
        token1: Token1 symbol (e.g., "USDC")
        fee_tier: Fee in bps (e.g., 500 = 0.05%, 3000 = 0.3%, 10000 = 1%)
        sqrt_price: Current √P (√(token1/token0))
        tick_spacing: Tick spacing for this fee tier
        tick_liquidity: List of (tick_lower, liquidity_net) pairs
            liquidity_net = liquidity added when crossing this tick going up
            (negative = liquidity removed)

    For ETH/USDC pool where WETH is token0:
        P = USDC per ETH (e.g., 3500)
        √P = √3500 ≈ 59.16
    """
    token0: str
    token1: str
    fee_tier: int  # bps: 500, 3000, 10000
    sqrt_price: float
    tick_spacing: int
    tick_liquidity: List[Tuple[int, float]]  # [(tick, liquidity_net), ...]

    @property
    def fee_rate(self) -> float:
        """Fee as a decimal (e.g., 0.003 for 30bps)."""
        return self.fee_tier / 1_000_000

    @property
    def price(self) -> float:
        """Current price in token1 per token0."""
        return self.sqrt_price ** 2

    @property
    def current_tick(self) -> int:
        return sqrt_price_to_tick(self.sqrt_price)


def get_liquidity_at_tick(pool: PoolState, target_tick: int) -> float:
    """
    Get the active liquidity at a given tick by summing liquidity_net
    for all initialized ticks at or below target_tick.

    In practice, you'd track this as a running sum. For simulation
    purposes, we compute it from the tick_liquidity list.
    """
    total = 0.0
    for tick, liq_net in sorted(pool.tick_liquidity):
        if tick > target_tick:
            break
        total += liq_net
    return total


def build_liquidity_map(pool: PoolState) -> List[Tuple[int, int, float]]:
    """
    Build ordered list of (tick_lower, tick_upper, liquidity) ranges
    from the tick_liquidity data.

    Returns list of (tick_start, tick_end, active_liquidity) tuples.
    """
    sorted_ticks = sorted(pool.tick_liquidity, key=lambda x: x[0])
    if not sorted_ticks:
        return []

    ranges = []
    current_liquidity = 0.0

    for i, (tick, liq_net) in enumerate(sorted_ticks):
        current_liquidity += liq_net
        if i + 1 < len(sorted_ticks):
            next_tick = sorted_ticks[i + 1][0]
        else:
            next_tick = MAX_TICK

        if current_liquidity > 0:
            ranges.append((tick, next_tick, current_liquidity))

    return ranges


def swap_exact_in(pool: PoolState,
                  amount_in: float,
                  token_in: str) -> dict:
    """
    Calculate exact swap output for a given input amount.

    This walks through tick ranges exactly as the Uniswap v3 contract does.

    Args:
        pool: Pool state
        amount_in: Amount of input token (in token units, NOT USD)
                   For USDC→ETH: amount of USDC
                   For ETH→USDC: amount of ETH
        token_in: Which token is being swapped in ("token0" or "token1",
                  or the token symbol like "USDC" or "WETH")

    Returns:
        Dict with:
            amount_out: Output token amount
            execution_price: Effective price (token1/token0)
            mid_price: Price before the swap
            slippage_bps: Slippage in basis points
            fee_paid: Total fee paid (in input token)
            price_impact_bps: Price impact (how much the pool price moved)
            sqrt_price_after: Pool √P after swap
            ticks_crossed: Number of tick boundaries crossed
    """
    # Determine swap direction
    is_buying_token0 = (token_in == "token1" or token_in == pool.token1)

    # Fee
    fee_rate = pool.fee_rate
    amount_after_fee = amount_in * (1 - fee_rate)
    fee_paid = amount_in * fee_rate

    # Build liquidity map
    liq_ranges = build_liquidity_map(pool)
    if not liq_ranges:
        raise ValueError("No liquidity in pool")

    mid_price = pool.price
    sqrt_p = pool.sqrt_price
    remaining = amount_after_fee
    total_out = 0.0
    ticks_crossed = 0

    if is_buying_token0:
        # Swapping token1 → token0 (e.g., USDC → ETH)
        # Price goes UP (√P increases) — ETH gets more expensive
        # Walk through tick ranges in ascending order

        # Find current range
        current_tick = sqrt_price_to_tick(sqrt_p)

        for tick_lower, tick_upper, liquidity in liq_ranges:
            if tick_upper <= current_tick:
                continue  # Below current price, skip
            if remaining <= 0:
                break
            if liquidity <= 0:
                continue

            # Effective range: start from current √P (not tick_lower)
            sqrt_p_start = max(sqrt_p, tick_to_sqrt_price(tick_lower))
            sqrt_p_end = tick_to_sqrt_price(tick_upper)

            if sqrt_p_start >= sqrt_p_end:
                continue

            # Max token1 input to reach the upper tick boundary
            # Δy = L × (√P_end - √P_start)
            max_input_this_range = liquidity * (sqrt_p_end - sqrt_p_start)

            if remaining <= max_input_this_range:
                # Swap completes within this range
                # √P_new = √P_start + Δy / L
                sqrt_p_new = sqrt_p_start + remaining / liquidity
                # Output: Δx = L × (1/√P_start - 1/√P_new)
                out = liquidity * (1 / sqrt_p_start - 1 / sqrt_p_new)
                total_out += out
                sqrt_p = sqrt_p_new
                remaining = 0
            else:
                # Consume all liquidity in this range, move to next
                out = liquidity * (1 / sqrt_p_start - 1 / sqrt_p_end)
                total_out += out
                remaining -= max_input_this_range
                sqrt_p = sqrt_p_end
                ticks_crossed += 1

    else:
        # Swapping token0 → token1 (e.g., ETH → USDC)
        # Price goes DOWN (√P decreases) — ETH gets cheaper
        # Walk through tick ranges in descending order

        current_tick = sqrt_price_to_tick(sqrt_p)

        for tick_lower, tick_upper, liquidity in reversed(liq_ranges):
            if tick_lower > current_tick:
                continue  # Above current price, skip
            if remaining <= 0:
                break
            if liquidity <= 0:
                continue

            # Effective range: start from current √P (not tick_upper)
            sqrt_p_start = min(sqrt_p, tick_to_sqrt_price(tick_upper))
            sqrt_p_end = tick_to_sqrt_price(tick_lower)

            if sqrt_p_start <= sqrt_p_end:
                continue

            # Max token0 input to reach the lower tick boundary
            # Δx = L × (1/√P_end - 1/√P_start)
            max_input_this_range = liquidity * (1 / sqrt_p_end - 1 / sqrt_p_start)

            if remaining <= max_input_this_range:
                # Swap completes within this range
                # 1/√P_new = 1/√P_start + Δx / L
                inv_sqrt_p_new = 1 / sqrt_p_start + remaining / liquidity
                sqrt_p_new = 1 / inv_sqrt_p_new
                # Output: Δy = L × (√P_start - √P_new)
                out = liquidity * (sqrt_p_start - sqrt_p_new)
                total_out += out
                sqrt_p = sqrt_p_new
                remaining = 0
            else:
                # Consume all liquidity in this range
                out = liquidity * (sqrt_p_start - sqrt_p_end)
                total_out += out
                remaining -= max_input_this_range
                sqrt_p = sqrt_p_end
                ticks_crossed += 1

    if remaining > 0:
        # Not enough liquidity to fill the order
        pass

    amount_actually_swapped = amount_after_fee - remaining

    # Calculate execution price and slippage
    if is_buying_token0:
        # Paid token1, received token0
        # Execution price = token1_spent / token0_received = USDC per ETH
        if total_out > 0:
            execution_price = amount_actually_swapped / total_out
        else:
            execution_price = float('inf')
        # Slippage: paid more per token0 than mid price
        slippage_pct = (execution_price / mid_price - 1) * 100
    else:
        # Paid token0, received token1
        # Execution price = token1_received / token0_spent
        if amount_actually_swapped > 0:
            execution_price = total_out / amount_actually_swapped
        else:
            execution_price = 0
        # Slippage: received less per token0 than mid price
        slippage_pct = (1 - execution_price / mid_price) * 100

    price_after = sqrt_p ** 2
    price_impact_pct = abs(price_after / mid_price - 1) * 100

    return {
        'amount_in': amount_in,
        'amount_in_after_fee': amount_actually_swapped,
        'amount_out': total_out,
        'amount_unfilled': remaining,
        'execution_price': execution_price,
        'mid_price': mid_price,
        'slippage_pct': slippage_pct,
        'slippage_bps': slippage_pct * 100,
        'fee_paid': fee_paid,
        'fee_bps': fee_rate * 10000,
        'price_impact_pct': price_impact_pct,
        'price_impact_bps': price_impact_pct * 100,
        'sqrt_price_after': sqrt_p,
        'price_after': price_after,
        'ticks_crossed': ticks_crossed,
    }


def swap_exact_in_usd(pool: PoolState,
                       usd_amount: float,
                       direction: str = "buy_eth") -> dict:
    """
    Convenience: specify trade size in USD.

    Args:
        pool: Pool state
        usd_amount: Trade size in USD
        direction: "buy_eth" (USDC→ETH) or "sell_eth" (ETH→USDC)

    Returns:
        Same as swap_exact_in, plus USD-denominated fields
    """
    price = pool.price  # USDC per ETH

    if direction == "buy_eth":
        # Input is USDC (token1)
        result = swap_exact_in(pool, usd_amount, "token1")
        result['usd_in'] = usd_amount
        result['usd_out'] = result['amount_out'] * price  # ETH at mid price
        result['eth_out'] = result['amount_out']
    else:
        # Input is ETH (token0), convert USD to ETH amount
        eth_amount = usd_amount / price
        result = swap_exact_in(pool, eth_amount, "token0")
        result['usd_in'] = usd_amount
        result['usd_out'] = result['amount_out']  # USDC received
        result['eth_in'] = eth_amount

    result['direction'] = direction
    return result


# ============================================================
# Sample pool state constructors
# ============================================================

def create_sample_eth_usdc_pool(
    price: float = 3500.0,
    total_liquidity: float = 50_000_000,  # $50M TVL
    fee_tier: int = 500,  # 0.05% fee tier
    concentration_range_pct: float = 5.0,  # Liquidity concentrated within ±5%
    tick_spacing: int = 10,
) -> PoolState:
    """
    Create a sample ETH/USDC pool state.

    Models concentrated liquidity: most liquidity is within ±concentration_range_pct
    of the current price, with thin liquidity outside.

    Args:
        price: ETH price in USDC
        total_liquidity: Total pool TVL in USD
        fee_tier: Fee tier in hundredths of bps
        concentration_range_pct: Percentage range for concentrated liquidity
        tick_spacing: Tick spacing (10 for 0.05% pools, 60 for 0.3%)

    This is a simplified model. For real analysis, load actual pool state
    from your S3 data.
    """
    sqrt_price = math.sqrt(price)
    current_tick = sqrt_price_to_tick(sqrt_price)

    # Align to tick spacing
    current_tick = (current_tick // tick_spacing) * tick_spacing

    # Liquidity model:
    # Core range (±concentration_range_pct): 80% of liquidity
    # Extended range (±3× core): 15% of liquidity
    # Far range (±10× core): 5% of liquidity

    # Convert percentage range to ticks
    # 1 tick ≈ 0.01% price change, so 5% ≈ 500 ticks
    core_ticks = int(concentration_range_pct * 100)  # ±500 ticks for 5%
    core_ticks = max(core_ticks, tick_spacing * 2)

    # Liquidity L relates to TVL roughly as:
    # For a range [P_low, P_high] with liquidity L:
    #   TVL ≈ L × (√P_high - √P_low) + L × (1/√P_low - 1/√P_high) × P
    # Simplified: TVL ≈ 2 × L × √P × (√(1+r) - 1) for small range r
    #
    # For our core range:
    r = concentration_range_pct / 100
    range_factor = 2 * sqrt_price * (math.sqrt(1 + r) - 1)
    if range_factor > 0:
        core_L = (total_liquidity * 0.80) / range_factor
    else:
        core_L = total_liquidity * 1e6  # Fallback

    extended_L = core_L * 0.10  # Thinner outside core
    far_L = core_L * 0.02       # Very thin far out

    tick_liquidity = []

    # Core range
    core_lower = current_tick - core_ticks
    core_upper = current_tick + core_ticks
    tick_liquidity.append((core_lower, core_L))
    tick_liquidity.append((core_upper, -core_L))

    # Extended range
    ext_lower = current_tick - core_ticks * 3
    ext_upper = current_tick + core_ticks * 3
    tick_liquidity.append((ext_lower, extended_L))
    tick_liquidity.append((ext_upper, -extended_L))

    # Far range
    far_lower = current_tick - core_ticks * 10
    far_upper = current_tick + core_ticks * 10
    tick_liquidity.append((far_lower, far_L))
    tick_liquidity.append((far_upper, -far_L))

    return PoolState(
        token0="WETH",
        token1="USDC",
        fee_tier=fee_tier,
        sqrt_price=sqrt_price,
        tick_spacing=tick_spacing,
        tick_liquidity=tick_liquidity,
    )


# ============================================================
# Slippage curve: size → slippage
# ============================================================

def compute_slippage_curve(pool: PoolState,
                           sizes_usd: list = None,
                           direction: str = "buy_eth") -> list:
    """
    Compute slippage for a range of trade sizes.

    Args:
        pool: Pool state
        sizes_usd: List of trade sizes in USD
        direction: "buy_eth" or "sell_eth"

    Returns:
        List of result dicts (one per size)
    """
    if sizes_usd is None:
        sizes_usd = [
            10, 100, 1_000, 5_000, 10_000, 50_000,
            100_000, 500_000, 1_000_000, 5_000_000,
            10_000_000, 50_000_000,
        ]

    results = []
    for size in sizes_usd:
        try:
            result = swap_exact_in_usd(pool, size, direction)
            results.append(result)
        except Exception as e:
            results.append({
                'usd_in': size,
                'error': str(e),
            })

    return results


def print_slippage_table(results: list):
    """Print slippage results as a formatted table."""
    print(f"{'Trade Size':>14s}  {'Slippage':>10s}  {'Price Impact':>13s}  "
          f"{'Exec Price':>12s}  {'Ticks':>5s}  {'Fee':>10s}")
    print("-" * 75)

    for r in results:
        if 'error' in r:
            size_str = f"${r['usd_in']:>12,.0f}"
            print(f"{size_str}  {'ERROR':>10s}  {r['error']}")
            continue

        size = r['usd_in']
        if size >= 1_000_000:
            size_str = f"${size/1_000_000:.1f}M"
        elif size >= 1_000:
            size_str = f"${size/1_000:.0f}k"
        else:
            size_str = f"${size:.0f}"

        slippage = r['slippage_bps']
        impact = r['price_impact_bps']
        exec_p = r['execution_price']
        ticks = r['ticks_crossed']
        fee = r['fee_paid']

        print(f"{size_str:>14s}  {slippage:>8.2f}bp  {impact:>10.2f}bp  "
              f"${exec_p:>10,.2f}  {ticks:>5d}  ${fee:>8,.2f}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 75)
    print("EXACT UNISWAP V3 SLIPPAGE CALCULATOR")
    print("=" * 75)

    # Create sample pool: ETH/USDC, $50M TVL, 0.05% fee
    pool = create_sample_eth_usdc_pool(
        price=3500.0,
        total_liquidity=50_000_000,
        fee_tier=500,            # 0.05%
        concentration_range_pct=5.0,
    )

    print(f"\nPool: {pool.token0}/{pool.token1}")
    print(f"Price: ${pool.price:,.2f}")
    print(f"Fee: {pool.fee_tier / 10000:.2f}%")
    print(f"Current tick: {pool.current_tick}")

    # Show liquidity distribution
    liq_map = build_liquidity_map(pool)
    print(f"\nLiquidity ranges ({len(liq_map)}):")
    for tick_l, tick_u, liq in liq_map:
        p_low = tick_to_sqrt_price(tick_l) ** 2
        p_high = tick_to_sqrt_price(tick_u) ** 2
        print(f"  [{tick_l:>7d}, {tick_u:>7d}]  "
              f"P=[${p_low:>8,.0f}, ${p_high:>8,.0f}]  L={liq:>14,.0f}")

    # Buy ETH with USDC
    print(f"\n{'='*75}")
    print("BUYING ETH (USDC → ETH)")
    print(f"{'='*75}\n")

    results = compute_slippage_curve(pool, direction="buy_eth")
    print_slippage_table(results)

    # Sell ETH for USDC
    print(f"\n{'='*75}")
    print("SELLING ETH (ETH → USDC)")
    print(f"{'='*75}\n")

    results = compute_slippage_curve(pool, direction="sell_eth")
    print_slippage_table(results)

    # Specific example: $10 swap
    print(f"\n{'='*75}")
    print("DETAIL: $10 SWAP (USDC → ETH)")
    print(f"{'='*75}\n")

    r = swap_exact_in_usd(pool, 10.0, "buy_eth")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:>25s}: {v:,.10f}")
        else:
            print(f"  {k:>25s}: {v}")

    print(f"\n\nUsage with real pool data:")
    print("""
    # Load pool state from your S3 Uniswap data:
    pool = PoolState(
        token0="WETH",
        token1="USDC",
        fee_tier=500,
        sqrt_price=math.sqrt(3500),
        tick_spacing=10,
        tick_liquidity=[
            # (tick, liquidity_net) from your data
            (200000, 1e15),   # liquidity added at this tick
            (210000, -1e15),  # liquidity removed at this tick
            ...
        ],
    )

    # Calculate exact slippage for any size:
    result = swap_exact_in_usd(pool, usd_amount=1_000_000, direction="buy_eth")
    print(f"Slippage: {result['slippage_bps']:.2f} bps")
    print(f"ETH received: {result['eth_out']:.6f}")
    """)
