# Realistic Delta-Neutral Funding Rate Simulator — Design Document

## The Strategy

Capture perpetual futures funding rate via a delta-neutral position:
- **Long ETH spot** (hedge)
- **Short ETH perp** on Hyperliquid (earns funding when rate > 0)
- Optionally use **Aave** to lever up the spot side

## The Core Economics Problem

You have capital `C` in USDC. To be **delta neutral**, you must hold equal ETH exposure on both sides:

```
Long X ETH spot  ←→  Short X ETH perp
```

Funding is earned on the short perp notional = `X × price`.

**The catch:** You need USDC for perp margin AND USDC to buy spot ETH. Both come from the same capital pool.

### Without Aave (Simple Mode)

Split capital into two buckets:
```
C = spot_budget + perp_margin

spot_budget → buy (spot_budget / price) ETH
perp_margin → deposit to Hyperliquid as margin
Short (spot_budget / price) ETH perp  (= same ETH quantity as spot → delta neutral)
```

**Key insight:** The short notional = spot_budget < C. You can never get 1× effective leverage without Aave.

**Effective leverage:**
```
L_eff = spot_budget / C = 1 - (perp_margin / C)
```

- If you split 50/50: L_eff = 0.5× (earn half the funding rate on your capital)
- If you split 80/20: L_eff = 0.8× (more notional, but thinner perp margin)
- Margin ratio on perp = perp_margin / spot_budget

**Optimal split** depends on:
- Minimum margin ratio you're comfortable with (e.g., 20% → 5× perp leverage)
- Expected price volatility (more vol → need more margin)

**Example:** $100k capital, 50/50 split:
- Buy 14.28 ETH at $3500 ($50k)
- Short 14.28 ETH perp ($50k notional), margin = $50k (100% margin ratio)
- Funding on $50k notional ≈ 0.01%/hr × $50k × 8760h = $43.8k/yr
- Yield on $100k capital = 43.8% × 0.5 = **21.9% APR** (not 43.8%!)

### With Aave (Leveraged Mode)

Use Aave to "recycle" the spot ETH as collateral and borrow more USDC:

```
C = perp_margin + aave_capital

Step 1: Buy (aave_capital / price) ETH
Step 2: Supply ETH to Aave
Step 3: Borrow USDC from Aave (at some LTV ratio r)
Step 4: Buy more ETH with borrowed USDC
Step 5: Supply new ETH to Aave
Step 6: Repeat steps 3-5 until target leverage reached

After looping:
  Total ETH value = aave_capital / (1 - r)
  Total Aave debt = aave_capital × r / (1 - r)

Step 7: Short (total_ETH / price) ETH perp, margin = perp_margin
```

**Delta neutral:** spot ETH = short perp ETH (same quantity on both sides)

**Effective leverage on total capital:**
```
L_eff = total_ETH_value / C = aave_capital / ((1 - r) × C)
```

**Aave health factor:**
```
HF = (total_ETH_value × liquidation_threshold) / debt
   = liquidation_threshold / r

For HF ≥ 1.2:  r ≤ 0.825/1.2 = 0.6875
For HF ≥ 1.5:  r ≤ 0.825/1.5 = 0.55
```

**Maximum leverage** (with E-mode off, LTV=80%):
```
L_max = aave_capital / ((1 - 0.80) × C)
      = (C - perp_margin) / (0.20 × C)
```

**Perp margin ratio:**
```
margin_ratio = perp_margin / total_ETH_value
             = perp_margin / (aave_capital / (1 - r))
```

### Working Example: $100k Capital, Target 2× Leverage

**Step 1: Solve for capital split**

Want: L_eff = 2.0, with HF ≥ 1.2

```
L_eff = (C - M) / ((1-r) × C) = 2.0
where M = perp_margin

HF = 0.825 / r ≥ 1.2  →  r ≤ 0.6875
```

Need: `(100k - M) / ((1-r) × 100k) = 2.0`

With target HF = 1.2 → r = 0.6875:
```
(100k - M) / (0.3125 × 100k) = 2.0
100k - M = 62,500
M = 37,500
```

**Step 2: Execute**
```
perp_margin = $37,500
aave_capital = $62,500

Buy $62,500 / $3,500 = 17.86 ETH
Supply to Aave → borrow at 68.75% LTV → buy more → loop...
Total ETH after loop = $62,500 / (1 - 0.6875) = $200,000 → 57.14 ETH
Total Aave debt = $200,000 - $62,500 = $137,500

Short 57.14 ETH perp on Hyperliquid, margin = $37,500
Margin ratio = $37,500 / $200,000 = 18.75%
HF = 0.825 / 0.6875 = 1.20
```

**Result:**
- Funding earned on $200k notional → 2× rate on $100k capital ✓
- Delta neutral: 57.14 ETH long = 57.14 ETH short ✓
- HF = 1.20 (acceptable) ✓
- Perp margin ratio = 18.75% (healthy) ✓

### Without Aave, Same Capital

```
perp_margin = $50,000  (50% split)
spot_budget = $50,000

Buy 14.28 ETH, short 14.28 ETH perp
Funding on $50k notional
L_eff = 0.5×
```

Funding earned is **4× less** than the leveraged version.

## What Happens When Price Moves

### ETH Goes Up (e.g., +20%)

**Perp side:**
- Short loses: 57.14 ETH × ($4200 - $3500) = -$40,000
- Margin was $37,500 → equity = -$2,500 → **LIQUIDATED** on HL before this

**Aave side:**
- Collateral worth: 57.14 × $4200 = $240,000 (up from $200k)
- Debt unchanged: $137,500
- HF = ($240,000 × 0.825) / $137,500 = 1.44 (improved!)

**The problem:** HL gets liquidated before the Aave side benefits.

**Solution:** Rebalancing:
- When HL margin ratio drops (say below 10%), top up margin by borrowing more from Aave
- This increases Aave debt → lowers HF
- Eventually if price keeps going up, Aave HF drops to 1.0 → Aave liquidation

### ETH Goes Down (e.g., -20%)

**Perp side:**
- Short gains: 57.14 × ($3500 - $2800) = +$40,000
- Margin = $37,500 + $40,000 = $77,500 (very healthy)

**Aave side:**
- Collateral worth: 57.14 × $2800 = $160,000
- Debt: $137,500
- HF = ($160,000 × 0.825) / $137,500 = 0.96 → **LIQUIDATED** on Aave!

**Solution:** Rebalancing:
- When Aave HF drops (say below 1.3), repay debt using excess HL margin
- Or reduce position size on both sides

## Implementation Plan

### 1. `HyperliquidAccount` class (existing, mostly correct)
- Tracks: margin, position size, entry price, unrealized PnL, funding, fees
- Methods: deposit, open_short, close_short, mark_to_market, receive_funding
- Liquidation: when equity / notional < 3% maintenance margin

### 2. `AavePosition` class (existing, mostly correct)
- Tracks: collateral ETH, debt USDC, interest accrual
- Methods: supply, borrow, repay, withdraw
- Health factor: (collateral × liq_threshold) / debt
- Liquidation: when HF < 1.0

### 3. `DeltaNeutralSimulator` class (needs rewrite)

**Initialization** — Given: capital C, leverage L, margin_fraction α

```
if mode == "simple":
    # No Aave
    spot_budget = C × (1 - α)
    perp_margin = C × α
    eth_amount = spot_budget / price

    Buy eth_amount ETH spot
    Short eth_amount ETH perp, margin = perp_margin

    L_eff = (1 - α)  # always < 1

elif mode == "leveraged":
    # With Aave looping
    # Given target L, solve for α and r
    # r = LTV used in Aave looping
    # Constraint: HF = liq_threshold / r ≥ target_hf

    perp_margin = α × C
    aave_capital = (1 - α) × C

    # Aave loop: total_spot_value = aave_capital / (1 - r)
    # L_eff = total_spot_value / C = (1-α) / (1-r)
    # Solve: r = 1 - (1-α)/L

    total_spot_value = C × L
    total_eth = total_spot_value / price
    aave_debt = total_spot_value - aave_capital

    Supply total_eth to Aave
    Borrow aave_debt USDC from Aave
    Short total_eth perp, margin = perp_margin
```

**Per-step simulation:**
```
for each hour:
    1. Update Aave collateral value (ETH price changed)
    2. Update HL unrealized PnL (ETH price changed)
    3. Check HL liquidation (margin_ratio < 3%)
    4. Check Aave liquidation (HF < 1.0)
    5. Receive funding payment (funding_rate × notional)
    6. Accrue Aave interest (borrow rate on debt, supply rate on collateral)
    7. Check if rebalance needed:
       a. HL margin too low → borrow from Aave, deposit to HL
       b. Aave HF too low → withdraw from HL, repay Aave debt
       c. Both critical → reduce position (close partial short + withdraw Aave)
    8. Record snapshot
```

**Rebalancing logic:**
```
HL_MARGIN_THRESHOLD = 0.08   # Rebalance when margin ratio < 8%
HL_MARGIN_TARGET = 0.15      # Target 15% after rebalance
AAVE_HF_THRESHOLD = 1.15     # Rebalance when HF < 1.15
AAVE_HF_TARGET = 1.30        # Target 1.30 after rebalance

if hl_margin_ratio < HL_MARGIN_THRESHOLD:
    # Need more USDC in HL
    top_up_needed = target_margin - current_equity

    # Source: borrow from Aave (if HF allows)
    aave_borrowable = collateral_value × max_ltv - debt
    borrow = min(top_up_needed, aave_borrowable × 0.9)

    if borrow > 0:
        aave.borrow(borrow)
        hl.deposit_margin(borrow)
    else:
        # Can't borrow more → reduce position
        reduce_position(fraction=0.2)

if aave_hf < AAVE_HF_THRESHOLD:
    # Need to reduce Aave debt
    target_debt = collateral_value × liq_threshold / AAVE_HF_TARGET
    repay_needed = debt - target_debt

    # Source: excess HL margin
    hl_excess = hl_equity - (notional × 0.08)  # keep 8% minimum
    repay = min(repay_needed, max(0, hl_excess))

    if repay > 0:
        hl.margin -= repay
        aave.repay(repay)
    else:
        # Not enough excess → reduce position
        reduce_position(fraction=0.2)

def reduce_position(fraction):
    reduce_eth = position_eth × fraction
    hl.close_short(reduce_eth)    # Frees margin
    aave.withdraw(reduce_eth)     # Frees collateral
    aave.repay(reduce_eth × price)  # Reduces debt
    spot_eth -= reduce_eth
```

### 4. Output / Tracking

Each timestep records:
```
{
    timestamp, eth_price,

    # Position state
    spot_eth, spot_value,
    short_eth, short_notional,

    # Hyperliquid
    hl_margin, hl_equity, hl_margin_ratio,
    hl_unrealized_pnl, hl_funding_this_step,
    hl_cumulative_funding, hl_cumulative_fees,

    # Aave
    aave_collateral_eth, aave_collateral_value,
    aave_debt, aave_hf, aave_ltv,
    aave_interest_this_step, aave_supply_yield_this_step,

    # Portfolio
    total_equity,  # = aave_equity + hl_equity + cash
    pnl, pnl_pct,
    effective_leverage,  # = short_notional / total_equity

    # Events
    rebalanced, rebalance_type,  # 'hl_topup', 'aave_repay', 'reduce'
    liquidated, liquidation_side,
}
```

### 5. Slippage Model (Future: Uniswap S3 Data)

Currently using flat fee (0.035% taker). Future enhancement:
- Load real Uniswap swap data from S3
- Build empirical curve: `swap_size → realized_slippage`
- Include MEV/sandwich attack costs
- Apply this to every trade (open, close, rebalance)

### 6. Input Data Format

```python
price_data = pd.DataFrame({
    'timestamp': [...],           # datetime
    'eth_price': [...],           # USD price
    'funding_rate': [...],        # per-hour rate (e.g., 0.0001 = 0.01%)
    'borrow_apy': [...],          # Aave USDC borrow APY (annualized)
    'supply_apy': [...],          # Aave WETH supply APY (annualized)
    'hours_elapsed': 1.0,         # time between rows (default 1h)
})
```

### 7. Key Questions / Design Decisions

1. **α (margin fraction):** Should this be a user parameter or auto-computed?
   - Proposal: Auto-compute from target leverage and HF constraint
   - User can override with explicit `margin_fraction` parameter

2. **Rebalance thresholds:** Fixed or adaptive?
   - Start with fixed thresholds, allow user to tune

3. **Funding frequency:** Hyperliquid pays hourly but conceptually settles every 8h.
   - Model hourly since our data is hourly

4. **Position reduction:** When both HL and Aave are near liquidation, how aggressively to deleverage?
   - Start with 20% reduction per step, cap at 3 reductions per event
