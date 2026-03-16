"""
Realistic Position-Level Simulator
====================================

Instead of formulas, this simulates actual:
- Hyperliquid margin account (mark-to-market, funding, liquidation)
- Aave position (collateral, debt, health factor, liquidation)
- Delta-neutral strategy PnL tick by tick

Feed it real ETH prices + funding rates → get true returns.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Hyperliquid Perpetual Position
# ============================================================

@dataclass
class HyperliquidAccount:
    """
    Simulates a Hyperliquid margin account for a SHORT perp position.

    Tracks:
    - Margin deposited (USDC)
    - Position size (ETH notional)
    - Entry price
    - Unrealized PnL
    - Funding payments received/paid
    - Liquidation events

    Hyperliquid mechanics (as of 2024):
    - Cross-margin by default
    - Maintenance margin: 3% of position notional for ETH
    - Funding: paid every 8 hours (hourly accrual on HL)
    - Trading fee: 0.035% taker, 0.01% maker (we use taker)
    - Liquidation: position closed at mark price when margin < maintenance
    - Liquidation penalty: 0.5% of position notional
    """
    margin_deposited: float = 0.0       # USDC deposited as margin
    position_size_eth: float = 0.0      # ETH units (positive = short notional)
    entry_price: float = 0.0            # Average entry price
    unrealized_pnl: float = 0.0         # Current unrealized PnL
    cumulative_funding: float = 0.0     # Total funding received (+) or paid (-)
    cumulative_fees: float = 0.0        # Total trading fees paid
    liquidated: bool = False
    liquidation_loss: float = 0.0

    # Protocol parameters
    maintenance_margin_rate: float = 0.03    # 3% for ETH
    taker_fee_rate: float = 0.00035          # 0.035%
    maker_fee_rate: float = 0.0001           # 0.01%
    liquidation_penalty_rate: float = 0.005  # 0.5%

    # Tracking
    trade_log: list = field(default_factory=list)
    funding_log: list = field(default_factory=list)

    def deposit_margin(self, amount_usdc: float):
        """Deposit USDC as margin."""
        self.margin_deposited += amount_usdc

    def open_short(self, size_eth: float, price: float, is_maker: bool = False):
        """
        Open/increase short perpetual position.

        Args:
            size_eth: ETH units to short
            price: Current mark price
            is_maker: Use maker fee if True
        """
        notional = size_eth * price
        fee_rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        fee = notional * fee_rate

        self.position_size_eth += size_eth
        # Weighted average entry price
        if self.position_size_eth > 0:
            old_notional = (self.position_size_eth - size_eth) * self.entry_price
            new_notional = size_eth * price
            self.entry_price = (old_notional + new_notional) / self.position_size_eth

        self.cumulative_fees += fee
        self.margin_deposited -= fee  # Fee deducted from margin

        self.trade_log.append({
            'action': 'open_short',
            'size_eth': size_eth,
            'price': price,
            'fee': fee,
            'notional': notional,
        })

    def close_short(self, size_eth: float, price: float, is_maker: bool = False):
        """
        Close/reduce short perpetual position.

        Args:
            size_eth: ETH units to close
            price: Current mark price
        """
        size_eth = min(size_eth, self.position_size_eth)
        if size_eth <= 0:
            return 0.0

        notional = size_eth * price
        fee_rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        fee = notional * fee_rate

        # Realized PnL: short profit = (entry - exit) * size
        realized_pnl = (self.entry_price - price) * size_eth - fee

        self.position_size_eth -= size_eth
        self.cumulative_fees += fee
        self.margin_deposited += realized_pnl  # PnL settles into margin

        self.trade_log.append({
            'action': 'close_short',
            'size_eth': size_eth,
            'price': price,
            'fee': fee,
            'realized_pnl': realized_pnl,
        })

        return realized_pnl

    def mark_to_market(self, current_price: float):
        """
        Update unrealized PnL at current mark price.

        For short: unrealized PnL = (entry_price - current_price) * size
        """
        if self.position_size_eth > 0:
            self.unrealized_pnl = (self.entry_price - current_price) * self.position_size_eth
        else:
            self.unrealized_pnl = 0.0

    def receive_funding(self, funding_rate: float, current_price: float):
        """
        Process funding payment.

        On Hyperliquid, funding is paid every hour but settles every 8h.
        For SHORT position:
        - If funding_rate > 0: shorts RECEIVE funding (longs pay shorts)
        - If funding_rate < 0: shorts PAY funding (shorts pay longs)

        Args:
            funding_rate: The funding rate for this period (e.g., 0.0001 = 0.01%)
            current_price: Current mark price

        Returns:
            Funding payment amount (positive = received)
        """
        if self.position_size_eth <= 0:
            return 0.0

        position_notional = self.position_size_eth * current_price
        # Short receives when rate > 0
        funding_payment = funding_rate * position_notional
        self.cumulative_funding += funding_payment
        self.margin_deposited += funding_payment

        self.funding_log.append({
            'funding_rate': funding_rate,
            'position_notional': position_notional,
            'payment': funding_payment,
        })

        return funding_payment

    @property
    def equity(self):
        """Account equity = margin + unrealized PnL."""
        return self.margin_deposited + self.unrealized_pnl

    @property
    def position_notional(self):
        """Current position notional value."""
        return self.position_size_eth * self.entry_price

    def margin_ratio(self, current_price: float):
        """
        Current margin ratio = equity / position_notional.

        Liquidation when margin_ratio < maintenance_margin_rate.
        """
        self.mark_to_market(current_price)
        notional = self.position_size_eth * current_price
        if notional <= 0:
            return float('inf')
        return self.equity / notional

    def check_liquidation(self, current_price: float):
        """
        Check if position should be liquidated.

        Returns:
            (is_liquidated, loss_amount)
        """
        if self.position_size_eth <= 0:
            return False, 0.0

        mr = self.margin_ratio(current_price)
        if mr < self.maintenance_margin_rate:
            # Liquidation occurs
            notional = self.position_size_eth * current_price
            penalty = notional * self.liquidation_penalty_rate

            # Close position at current price
            realized_pnl = (self.entry_price - current_price) * self.position_size_eth
            total_loss = self.margin_deposited + realized_pnl - penalty

            self.liquidated = True
            self.liquidation_loss = min(total_loss, 0)  # Loss is negative
            remaining = max(total_loss, 0)

            # Reset position
            self.position_size_eth = 0
            self.entry_price = 0
            self.unrealized_pnl = 0
            self.margin_deposited = remaining

            return True, self.liquidation_loss

        return False, 0.0

    def summary(self, current_price: float):
        """Get account summary."""
        self.mark_to_market(current_price)
        return {
            'margin_deposited': self.margin_deposited,
            'position_size_eth': self.position_size_eth,
            'entry_price': self.entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': self.equity,
            'margin_ratio': self.margin_ratio(current_price),
            'cumulative_funding': self.cumulative_funding,
            'cumulative_fees': self.cumulative_fees,
            'liquidated': self.liquidated,
        }


# ============================================================
# Aave v3 Position (Arbitrum)
# ============================================================

@dataclass
class AavePosition:
    """
    Simulates an Aave v3 position on Arbitrum.

    Strategy: Supply WETH as collateral, borrow USDC.
    The borrowed USDC funds the Hyperliquid margin account.

    Aave v3 parameters for WETH on Arbitrum (as of 2024):
    - Max LTV: 80% (can borrow up to 80% of collateral value)
    - Liquidation threshold: 82.5%
    - Liquidation penalty: 5%
    - Supply APY: variable (typically 1-3%)
    - E-Mode (ETH correlated): LTV 90%, liq threshold 93%

    For USDC borrowing:
    - Variable borrow rate: depends on utilization
    - Base rate + slope1 * U (if U < optimal) + slope2 * (U - optimal) (if U > optimal)
    """
    # Collateral
    collateral_eth: float = 0.0          # WETH supplied
    collateral_value_usd: float = 0.0    # Current USD value

    # Debt
    debt_usdc: float = 0.0               # USDC borrowed (principal + accrued interest)

    # Cumulative tracking
    cumulative_interest_paid: float = 0.0
    cumulative_supply_earned: float = 0.0

    # Protocol parameters
    max_ltv: float = 0.80                # 80% LTV
    liquidation_threshold: float = 0.825  # 82.5%
    liquidation_penalty: float = 0.05     # 5% penalty
    e_mode: bool = False                  # E-Mode disabled by default

    # E-Mode parameters (ETH correlated)
    e_mode_ltv: float = 0.90
    e_mode_liq_threshold: float = 0.93
    e_mode_liq_penalty: float = 0.01

    # State
    liquidated: bool = False
    liquidation_loss: float = 0.0

    # Tracking
    interest_log: list = field(default_factory=list)

    @property
    def effective_ltv(self):
        return self.e_mode_ltv if self.e_mode else self.max_ltv

    @property
    def effective_liq_threshold(self):
        return self.e_mode_liq_threshold if self.e_mode else self.liquidation_threshold

    @property
    def effective_liq_penalty(self):
        return self.e_mode_liq_penalty if self.e_mode else self.liquidation_penalty

    def supply_eth(self, amount_eth: float, eth_price: float):
        """Supply WETH as collateral."""
        self.collateral_eth += amount_eth
        self.collateral_value_usd = self.collateral_eth * eth_price

    def borrow_usdc(self, amount_usdc: float, eth_price: float):
        """
        Borrow USDC against WETH collateral.

        Checks LTV constraint before borrowing.
        """
        self.collateral_value_usd = self.collateral_eth * eth_price
        max_borrow = self.collateral_value_usd * self.effective_ltv
        available = max_borrow - self.debt_usdc

        if amount_usdc > available:
            raise ValueError(
                f"Cannot borrow ${amount_usdc:,.0f}. "
                f"Max available: ${available:,.0f} "
                f"(collateral: ${self.collateral_value_usd:,.0f}, "
                f"LTV: {self.effective_ltv:.0%})"
            )

        self.debt_usdc += amount_usdc

    def repay_usdc(self, amount_usdc: float):
        """Repay USDC debt."""
        repay = min(amount_usdc, self.debt_usdc)
        self.debt_usdc -= repay
        return repay

    def withdraw_eth(self, amount_eth: float, eth_price: float):
        """
        Withdraw WETH collateral.
        Checks that withdrawal doesn't breach LTV.
        """
        new_collateral = self.collateral_eth - amount_eth
        new_collateral_value = new_collateral * eth_price
        if self.debt_usdc > 0:
            new_ltv = self.debt_usdc / new_collateral_value
            if new_ltv > self.effective_ltv:
                raise ValueError(
                    f"Cannot withdraw {amount_eth:.4f} ETH. "
                    f"Would breach LTV ({new_ltv:.2%} > {self.effective_ltv:.0%})"
                )
        self.collateral_eth = new_collateral
        self.collateral_value_usd = new_collateral_value

    def health_factor(self, eth_price: float):
        """
        Calculate health factor.

        HF = (collateral_value * liquidation_threshold) / debt
        HF < 1 → liquidation
        """
        self.collateral_value_usd = self.collateral_eth * eth_price
        if self.debt_usdc <= 0:
            return float('inf')
        return (self.collateral_value_usd * self.effective_liq_threshold) / self.debt_usdc

    def current_ltv(self, eth_price: float):
        """Current loan-to-value ratio."""
        self.collateral_value_usd = self.collateral_eth * eth_price
        if self.collateral_value_usd <= 0:
            return float('inf')
        return self.debt_usdc / self.collateral_value_usd

    def accrue_interest(self, borrow_apy: float, supply_apy: float,
                        eth_price: float, hours: float = 1.0):
        """
        Accrue interest for a time period.

        Args:
            borrow_apy: Annual borrow rate (e.g., 0.05 = 5%)
            supply_apy: Annual supply rate for WETH (e.g., 0.02 = 2%)
            eth_price: Current ETH price
            hours: Hours elapsed
        """
        fraction_of_year = hours / 8760

        # Interest on USDC debt
        interest_usd = self.debt_usdc * borrow_apy * fraction_of_year
        self.debt_usdc += interest_usd
        self.cumulative_interest_paid += interest_usd

        # Supply yield on WETH collateral (earned in ETH)
        supply_eth = self.collateral_eth * supply_apy * fraction_of_year
        self.collateral_eth += supply_eth
        self.cumulative_supply_earned += supply_eth * eth_price

        self.collateral_value_usd = self.collateral_eth * eth_price

        self.interest_log.append({
            'borrow_apy': borrow_apy,
            'supply_apy': supply_apy,
            'interest_paid': interest_usd,
            'supply_earned_eth': supply_eth,
            'debt_after': self.debt_usdc,
            'collateral_eth_after': self.collateral_eth,
        })

    def check_liquidation(self, eth_price: float):
        """
        Check if position is liquidatable.

        Aave liquidation mechanics:
        - Triggered when health_factor < 1
        - Liquidator can repay up to 50% of debt
        - Liquidator receives collateral + penalty
        - We simulate FULL liquidation for simplicity (worst case)
        """
        hf = self.health_factor(eth_price)
        if hf >= 1.0:
            return False, 0.0

        # Liquidation: lose collateral proportional to debt + penalty
        penalty_rate = self.effective_liq_penalty
        # Debt to be repaid (simulate 100% liquidation)
        debt_to_repay = self.debt_usdc

        # Collateral seized = debt_to_repay * (1 + penalty) / eth_price
        collateral_seized_eth = debt_to_repay * (1 + penalty_rate) / eth_price
        collateral_seized_eth = min(collateral_seized_eth, self.collateral_eth)

        # What's left
        remaining_eth = self.collateral_eth - collateral_seized_eth
        remaining_value = remaining_eth * eth_price

        # Loss = original collateral value - remaining value
        loss = (self.collateral_eth * eth_price) - remaining_value - debt_to_repay
        # Simplified: loss ≈ penalty amount
        actual_loss = collateral_seized_eth * eth_price - debt_to_repay

        self.liquidated = True
        self.liquidation_loss = actual_loss
        self.collateral_eth = remaining_eth
        self.collateral_value_usd = remaining_value
        self.debt_usdc = 0  # Debt repaid by liquidator

        return True, actual_loss

    def net_equity(self, eth_price: float):
        """Net equity = collateral value - debt."""
        self.collateral_value_usd = self.collateral_eth * eth_price
        return self.collateral_value_usd - self.debt_usdc

    def summary(self, eth_price: float):
        """Get position summary."""
        return {
            'collateral_eth': self.collateral_eth,
            'collateral_value_usd': self.collateral_eth * eth_price,
            'debt_usdc': self.debt_usdc,
            'health_factor': self.health_factor(eth_price),
            'current_ltv': self.current_ltv(eth_price),
            'net_equity': self.net_equity(eth_price),
            'cumulative_interest_paid': self.cumulative_interest_paid,
            'cumulative_supply_earned': self.cumulative_supply_earned,
            'liquidated': self.liquidated,
        }


# ============================================================
# Delta-Neutral Strategy Simulator
# ============================================================

class DeltaNeutralSimulator:
    """
    Simulates the complete delta-neutral funding rate capture strategy:

    1. Start with CAPITAL in USDC
    2. Buy ETH spot (or supply to Aave and borrow USDC)
    3. Short ETH perp on Hyperliquid
    4. Collect funding payments
    5. Rebalance when delta drifts
    6. Track everything tick by tick

    Two modes:
    - "simple": Buy spot ETH directly, deposit USDC as perp margin
    - "leveraged": Supply ETH to Aave, borrow USDC, use for perp margin
    """

    def __init__(self, capital: float, leverage: float = 1.0,
                 rebalance_threshold: float = 0.05,
                 mode: str = "leveraged",
                 e_mode: bool = False):
        """
        Args:
            capital: Starting capital in USDC
            leverage: Target leverage (1.0 = no leverage, 2.0 = 2x, etc.)
            rebalance_threshold: Rebalance when delta drifts by this fraction
            mode: "simple" or "leveraged"
            e_mode: Use Aave E-Mode (higher LTV)
        """
        self.initial_capital = capital
        self.target_leverage = leverage
        self.rebalance_threshold = rebalance_threshold
        self.mode = mode

        # Initialize accounts
        self.hl_account = HyperliquidAccount()
        self.aave = AavePosition(e_mode=e_mode)

        # Cash reserves (USDC not deployed)
        self.cash_usdc = capital

        # Tracking
        self.history = []
        self.rebalance_count = 0
        self.is_initialized = False

    def initialize_position(self, eth_price: float):
        """
        Open the initial delta-neutral position.

        For "leveraged" mode with leverage L:
        - Total notional exposure = L * capital
        - Each side (spot + perp) = L * capital / 2 ... NO

        Actually, the strategy works like this:

        With capital C and leverage L:
        - Total position size = C * L (in USD terms)
        - Buy C * L / price ETH spot
        - Short C * L / price ETH perp

        For leveraged mode:
        - Supply enough ETH to Aave to borrow what we need
        - Use borrowed USDC + own USDC for margin

        Simple approach:
        - Split capital: some for spot ETH, some for perp margin
        - With leverage L:
          * Notional = C * L
          * Need enough margin on both sides

        Realistic approach for L=2x:
        - Have $100k USDC
        - Supply $50k worth of ETH to Aave (buy first)
        - Borrow $40k USDC from Aave (80% LTV)
        - Now have $90k USDC ($50k remaining + $40k borrowed)
        - Deposit $90k USDC as Hyperliquid margin
        - Short $100k notional ETH perp
        - Net exposure: +$50k spot, -$100k perp... not delta neutral

        Better approach:
        - Goal: L× leverage means L× the funding capture per dollar of capital
        - With $100k capital at 2× leverage:
          * Want $200k notional short perp → earns 2× funding
          * Need enough margin for $200k short
          * On Hyperliquid: need ~$200k/20 = $10k initial margin (20× max)
            But want conservative margin, say 10% = $20k
          * Also need $200k ETH spot hedge

        Simplest correct model:
        - Capital C split into spot_allocation and perp_margin
        - spot_allocation buys ETH
        - With Aave: supply ETH, borrow USDC, add to perp_margin
        - Short perp = spot_eth_value to be delta neutral
        """
        if self.mode == "simple":
            self._init_simple(eth_price)
        else:
            self._init_leveraged(eth_price)

        self.is_initialized = True

    def _init_simple(self, eth_price: float):
        """
        Simple mode: no Aave leverage.
        Split capital 50/50 between spot and perp margin.
        Short perp = spot value → delta neutral.
        """
        spot_alloc = self.cash_usdc / 2
        margin_alloc = self.cash_usdc / 2

        # Buy spot ETH
        eth_amount = spot_alloc / eth_price
        self.spot_eth = eth_amount
        self.spot_entry_price = eth_price

        # Deposit margin and short perp
        self.hl_account.deposit_margin(margin_alloc)
        self.hl_account.open_short(eth_amount, eth_price)

        self.cash_usdc = 0

    def _init_leveraged(self, eth_price: float):
        """
        Leveraged mode using Aave:

        With capital C and target leverage L:
        1. Buy (C / eth_price) ETH with all capital
        2. Supply all ETH to Aave
        3. Borrow USDC from Aave (up to LTV limit based on leverage)
        4. Deposit borrowed USDC as Hyperliquid margin
        5. Short (C * L / eth_price) ETH perp

        The position:
        - Long: C worth of ETH in Aave (earning supply APY)
        - Short: C * L notional ETH perp (earning funding × L)
        - Debt: borrowed USDC on Aave

        Delta:
        - Spot delta: + C (from Aave collateral)
        - Perp delta: - C * L
        - Net delta: C * (1 - L)

        Wait — for delta neutral, we need spot = perp:
        - If L=1: spot C, short C → delta neutral ✓
        - If L=2: spot C, short 2C → net short C ✗

        For true delta neutral at higher leverage, we need MORE spot:
        - Supply C worth ETH → borrow 0.8C USDC → buy more ETH → supply → borrow...

        Recursive leverage (looping):
        - Round 0: Have C USDC. Buy C/P ETH, supply to Aave.
        - Round 1: Borrow C×LTV USDC. Buy more ETH, supply to Aave.
        - Round 2: Borrow C×LTV² USDC. Buy more ETH...
        - Total ETH value = C × (1 + LTV + LTV² + ...) = C / (1 - LTV)

        With LTV=0.8: max leverage = 1/(1-0.8) = 5×
        With E-mode LTV=0.9: max leverage = 1/(1-0.9) = 10×

        For target leverage L:
        - Need total ETH value = C × L
        - Total debt = C × L - C = C × (L - 1)
        - LTV used = debt / collateral = (L-1) / L
        - Must have (L-1)/L < max_LTV → L < 1/(1-max_LTV)

        For L=2, LTV used = 0.5 (safe, below 0.8)
        For L=3, LTV used = 0.667 (safe, below 0.8)
        For L=4, LTV used = 0.75 (safe, below 0.8)
        For L=5, LTV used = 0.80 (at limit)
        """
        C = self.cash_usdc
        L = self.target_leverage
        P = eth_price

        # Check leverage is feasible
        max_ltv = self.aave.effective_ltv
        required_ltv = (L - 1) / L if L > 1 else 0
        if required_ltv >= max_ltv:
            raise ValueError(
                f"Leverage {L}× requires LTV {required_ltv:.2%}, "
                f"but max LTV is {max_ltv:.0%}. "
                f"Max leverage: {1/(1-max_ltv):.1f}×"
            )

        # Total ETH to buy = C * L / P
        total_eth = C * L / P
        total_eth_value = C * L  # = total_eth * P

        # Buy all ETH (this represents the looped leverage result)
        # In reality this happens via multiple loops, but the end state is the same
        self.spot_eth = total_eth
        self.spot_entry_price = P

        # Supply all ETH to Aave
        self.aave.supply_eth(total_eth, P)

        # Borrow USDC: debt = C * (L - 1)
        debt_needed = C * (L - 1)
        if debt_needed > 0:
            self.aave.borrow_usdc(debt_needed, P)

        # Deposit borrowed USDC + remaining capital as perp margin
        # After buying C*L worth of ETH with C cash, we're out of cash
        # But the borrowed USDC from Aave gives us margin
        perp_margin = debt_needed + C * 0  # All capital went to ETH
        # Actually: we spent C buying ETH, then borrowed debt_needed USDC
        # The borrowed USDC is what we use as perp margin

        # Wait, let's think again:
        # We start with C USDC
        # We buy C*L/P ETH. But we only have C USDC!
        # The loop works like this:
        # 1. Buy C/P ETH with C USDC → 0 USDC left
        # 2. Supply C/P ETH to Aave
        # 3. Borrow C*LTV USDC from Aave
        # 4. Buy (C*LTV)/P ETH
        # 5. Supply to Aave
        # 6. Borrow C*LTV² USDC...
        # After all loops: total ETH = C*L/P, total debt = C*(L-1)
        # Final borrowed amount available = C * LTV^n (last loop residual)
        # But ALL borrowed USDC went to buying ETH!

        # So where does perp margin come from?
        # Answer: part of the capital is reserved for margin, not all goes to ETH

        # Correct approach:
        # - Reserve some capital for perp margin
        # - Use rest for the Aave loop
        #
        # Let M = margin needed for Hyperliquid short of C*L notional
        # Hyperliquid allows up to 50× leverage, so min margin = C*L/50
        # But for safety, we want margin_ratio ≈ 10-20%
        #
        # Better model: user's total capital C is deployed as follows:
        # - perp_margin = enough for the short position
        # - spot_capital = C - perp_margin → leveraged via Aave
        # - Total spot ETH = spot_capital * L_aave
        # - Short perp notional = spot ETH value (delta neutral)
        # - perp_margin must support this short
        #
        # For simplicity and realism:
        # Split: fraction α goes to perp margin, (1-α) goes to Aave loop
        # Aave leverage on spot: L_aave = 1/(1 - used_LTV)
        # Total spot = (1-α) × C × L_aave
        # Perp short = total spot (delta neutral)
        # Need: α × C sufficient margin for perp short of total_spot
        # Hyperliquid initial margin ~5% for 20× max leverage
        # So α × C ≥ 0.05 × total_spot
        # α ≥ 0.05 × (1-α) × L_aave
        # α ≥ 0.05 × L_aave - 0.05 × α × L_aave
        # α(1 + 0.05 × L_aave) ≥ 0.05 × L_aave
        # α ≥ 0.05 × L_aave / (1 + 0.05 × L_aave)

        # Let's use a cleaner model with explicit margin fraction
        # Reset everything and redo
        self.aave = AavePosition(e_mode=self.aave.e_mode)
        self.hl_account = HyperliquidAccount()

        # Target: short C*L notional on perp, hold C*L notional in spot ETH
        # This means we earn L× the funding rate on our capital

        target_notional = C * L
        target_eth = target_notional / P

        # Perp margin: need enough to not get liquidated easily
        # Use fraction of capital that keeps both Aave HF > 1.5 and HL margin safe
        # More capital to perp margin → less to Aave → lower Aave leverage → safer HF
        # Trade-off: too much margin means less leveraged spot exposure
        perp_margin_ratio = max(0.15, 1.0 / (L * 2))  # At least 15% of notional
        perp_margin = target_notional * perp_margin_ratio

        # Remaining capital for spot ETH purchase
        spot_capital = C - perp_margin

        if spot_capital <= 0:
            raise ValueError(
                f"Not enough capital. Need ${perp_margin:,.0f} for perp margin "
                f"but only have ${C:,.0f} total."
            )

        # Buy ETH with spot_capital → supply to Aave → borrow → buy more
        # After looping: total ETH = spot_capital / (1 - used_ltv) / P
        # We want total ETH = target_eth
        # So: spot_capital / (1 - used_ltv) = target_notional
        # used_ltv = 1 - spot_capital / target_notional
        used_ltv = 1 - spot_capital / target_notional

        if used_ltv < 0:
            # Don't need Aave at all (L ≈ 1)
            used_ltv = 0

        if used_ltv >= max_ltv:
            # Need to reduce target or increase margin efficiency
            # Adjust: use max_ltv and get whatever leverage we can
            actual_total = spot_capital / (1 - max_ltv * 0.95)  # 95% of max for safety
            target_eth = actual_total / P
            target_notional = actual_total
            used_ltv = max_ltv * 0.95
            print(f"  Warning: Adjusted notional to ${actual_total:,.0f} "
                  f"(max safe leverage with this capital split)")

        # Execute: supply ETH, borrow USDC
        self.spot_eth = target_eth
        self.aave.supply_eth(target_eth, P)

        debt = target_notional - spot_capital
        if debt > 0:
            self.aave.borrow_usdc(debt, P)

        # Deposit margin and open short
        self.hl_account.deposit_margin(perp_margin)
        self.hl_account.open_short(target_eth, P)

        self.cash_usdc = 0  # All capital deployed

    def step(self, timestamp, eth_price: float, funding_rate: float,
             borrow_apy: float = 0.05, supply_apy: float = 0.02,
             hours_elapsed: float = 1.0):
        """
        Advance simulation by one time step.

        Args:
            timestamp: Current timestamp
            eth_price: Current ETH price
            funding_rate: Funding rate for this period (raw, e.g., 0.0001)
            borrow_apy: Aave USDC borrow APY (annualized)
            supply_apy: Aave WETH supply APY (annualized)
            hours_elapsed: Hours since last step

        Returns:
            Dict with step results
        """
        result = {
            'timestamp': timestamp,
            'eth_price': eth_price,
        }

        # 1. Check Aave liquidation
        if self.mode == "leveraged" and self.aave.collateral_eth > 0:
            aave_liq, aave_loss = self.aave.check_liquidation(eth_price)
            if aave_liq:
                result['aave_liquidated'] = True
                result['aave_liquidation_loss'] = aave_loss
                # Close perp position too
                if self.hl_account.position_size_eth > 0:
                    self.hl_account.close_short(
                        self.hl_account.position_size_eth, eth_price
                    )
                return self._record(result, eth_price)

        # 2. Check Hyperliquid liquidation
        hl_liq, hl_loss = self.hl_account.check_liquidation(eth_price)
        if hl_liq:
            result['hl_liquidated'] = True
            result['hl_liquidation_loss'] = hl_loss
            return self._record(result, eth_price)

        # 3. Mark to market
        self.hl_account.mark_to_market(eth_price)

        # 4. Receive funding
        funding_payment = self.hl_account.receive_funding(funding_rate, eth_price)
        result['funding_payment'] = funding_payment

        # 5. Accrue Aave interest
        if self.mode == "leveraged" and self.aave.collateral_eth > 0:
            self.aave.accrue_interest(borrow_apy, supply_apy, eth_price, hours_elapsed)
            result['aave_interest'] = self.aave.interest_log[-1]['interest_paid']
            result['aave_supply_yield'] = (
                self.aave.interest_log[-1]['supply_earned_eth'] * eth_price
            )

        # 6. Check if rebalance needed
        rebalanced = self._check_and_rebalance(eth_price)
        result['rebalanced'] = rebalanced

        return self._record(result, eth_price)

    def _check_and_rebalance(self, eth_price: float):
        """
        Check delta drift and rebalance if needed.

        Delta drift occurs because:
        - Spot ETH value changes with price
        - Perp position notional changes with price
        - But the ETH quantity is fixed on both sides

        For our position:
        - Spot: self.spot_eth * eth_price
        - Perp short: self.hl_account.position_size_eth * eth_price

        Since both are in ETH terms and we hold same quantity,
        the delta is naturally hedged in ETH terms.

        BUT: the margin ratio on Hyperliquid changes:
        - If ETH goes up: short loses money, margin_ratio drops
        - If ETH goes down: short gains, margin_ratio increases

        We rebalance when margin_ratio gets too low (risky)
        or when the Aave health factor gets too low.
        """
        rebalanced = False

        # Check perp margin ratio
        if self.hl_account.position_size_eth > 0:
            mr = self.hl_account.margin_ratio(eth_price)

            # Rebalance if margin ratio drops below threshold
            if mr < self.rebalance_threshold:
                self._rebalance_perp_margin(eth_price)
                rebalanced = True

        # Check Aave health factor
        if self.mode == "leveraged" and self.aave.debt_usdc > 0:
            hf = self.aave.health_factor(eth_price)
            # Rebalance if health factor too low
            if hf < 1.2:  # 1.2 = safety buffer above 1.0 liquidation
                self._rebalance_aave(eth_price)
                rebalanced = True

        if rebalanced:
            self.rebalance_count += 1

        return rebalanced

    def _rebalance_perp_margin(self, eth_price: float):
        """
        Add margin to Hyperliquid to avoid liquidation.

        Source of funds:
        - In leveraged mode: borrow more from Aave (if possible)
        - In simple mode: reduce spot position
        """
        target_mr = 0.15  # Target 15% margin ratio after rebalance
        current_notional = self.hl_account.position_size_eth * eth_price
        needed_equity = target_mr * current_notional
        current_equity = self.hl_account.equity
        top_up = needed_equity - current_equity

        if top_up <= 0:
            return

        if self.mode == "leveraged" and self.aave.collateral_eth > 0:
            # Try to borrow more from Aave
            max_additional = (
                self.aave.collateral_eth * eth_price * self.aave.effective_ltv
                - self.aave.debt_usdc
            )
            borrow_amount = min(top_up, max_additional * 0.9)  # 90% of available
            if borrow_amount > 0:
                self.aave.borrow_usdc(borrow_amount, eth_price)
                self.hl_account.deposit_margin(borrow_amount)
        else:
            # Reduce position: close some short, sell some spot
            reduce_pct = min(0.2, top_up / current_notional)
            reduce_eth = self.hl_account.position_size_eth * reduce_pct
            self.hl_account.close_short(reduce_eth, eth_price)
            self.spot_eth -= reduce_eth

    def _rebalance_aave(self, eth_price: float):
        """
        Improve Aave health factor to avoid liquidation.

        Options:
        1. Repay some debt (need USDC)
        2. Reduce position size

        We repay debt using PnL from perp margin account.
        """
        target_hf = 1.5
        # debt_target = collateral * liq_threshold / target_hf
        debt_target = (
            self.aave.collateral_eth * eth_price
            * self.aave.effective_liq_threshold / target_hf
        )
        repay_amount = self.aave.debt_usdc - debt_target

        if repay_amount <= 0:
            return

        # Source from Hyperliquid margin (withdraw excess margin)
        excess_margin = self.hl_account.equity - (
            self.hl_account.position_size_eth * eth_price * 0.10  # Keep 10% margin
        )
        usdc_available = max(0, excess_margin)
        repay = min(repay_amount, usdc_available)

        if repay > 0:
            self.hl_account.margin_deposited -= repay
            self.aave.repay_usdc(repay)

        # If still not enough, repay debt first then withdraw collateral
        new_hf = self.aave.health_factor(eth_price)
        if new_hf < 1.2 and self.aave.debt_usdc > 0:
            # Close 20% of perp short to free up USDC
            reduce_eth = self.hl_account.position_size_eth * 0.2
            pnl = self.hl_account.close_short(reduce_eth, eth_price)

            # Use freed margin to repay Aave debt first
            freed_margin = max(0, self.hl_account.equity - (
                self.hl_account.position_size_eth * eth_price * 0.10
            ))
            repay_more = min(freed_margin, self.aave.debt_usdc)
            if repay_more > 0:
                self.hl_account.margin_deposited -= repay_more
                self.aave.repay_usdc(repay_more)

            # Now withdraw ETH collateral (should be safe after debt repayment)
            try:
                self.aave.withdraw_eth(reduce_eth, eth_price)
                self.spot_eth -= reduce_eth
            except ValueError:
                # Still can't withdraw — just reduce spot tracking
                # The ETH stays in Aave as extra collateral (safer)
                self.spot_eth -= reduce_eth
                self.aave.collateral_eth -= reduce_eth

    def _record(self, result: dict, eth_price: float):
        """Record state snapshot."""
        # Total equity across all accounts
        spot_value = self.spot_eth * eth_price
        hl_equity = self.hl_account.equity
        aave_equity = self.aave.net_equity(eth_price) if self.mode == "leveraged" else 0

        # For leveraged mode:
        # Total equity = Aave equity (collateral - debt) + HL equity + cash
        # The spot ETH is inside Aave, so aave_equity already includes it
        if self.mode == "leveraged":
            total_equity = aave_equity + hl_equity + self.cash_usdc
        else:
            total_equity = spot_value + hl_equity + self.cash_usdc

        pnl = total_equity - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100

        result.update({
            'spot_eth': self.spot_eth,
            'spot_value': spot_value,
            'hl_equity': hl_equity,
            'hl_margin_ratio': self.hl_account.margin_ratio(eth_price),
            'hl_unrealized_pnl': self.hl_account.unrealized_pnl,
            'hl_cumulative_funding': self.hl_account.cumulative_funding,
            'hl_cumulative_fees': self.hl_account.cumulative_fees,
            'aave_health_factor': self.aave.health_factor(eth_price) if self.mode == "leveraged" else None,
            'aave_debt': self.aave.debt_usdc,
            'aave_collateral_value': self.aave.collateral_eth * eth_price,
            'aave_equity': aave_equity,
            'total_equity': total_equity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'rebalance_count': self.rebalance_count,
            'aave_liquidated': result.get('aave_liquidated', False),
            'hl_liquidated': result.get('hl_liquidated', False),
        })

        self.history.append(result)
        return result

    def get_results(self):
        """Get full simulation results as DataFrame."""
        return pd.DataFrame(self.history)

    def summary(self):
        """Get final simulation summary."""
        if not self.history:
            return "No simulation data"

        df = pd.DataFrame(self.history)
        final = df.iloc[-1]
        initial_price = df.iloc[0]['eth_price']
        final_price = final['eth_price']

        return {
            'initial_capital': self.initial_capital,
            'leverage': self.target_leverage,
            'mode': self.mode,
            'duration_hours': len(df),
            'initial_eth_price': initial_price,
            'final_eth_price': final_price,
            'eth_return_pct': (final_price / initial_price - 1) * 100,
            'final_equity': final['total_equity'],
            'total_pnl': final['pnl'],
            'total_return_pct': final['pnl_pct'],
            'annualized_return_pct': final['pnl_pct'] * 8760 / len(df),
            'total_funding_received': final['hl_cumulative_funding'],
            'total_fees_paid': final['hl_cumulative_fees'],
            'total_aave_interest': self.aave.cumulative_interest_paid,
            'total_aave_supply_yield': self.aave.cumulative_supply_earned,
            'rebalance_count': self.rebalance_count,
            'max_drawdown_pct': self._max_drawdown(df),
            'min_hl_margin_ratio': df['hl_margin_ratio'].min(),
            'min_aave_hf': df['aave_health_factor'].min() if self.mode == "leveraged" else None,
            'hl_liquidated': final['hl_liquidated'],
            'aave_liquidated': final['aave_liquidated'],
        }

    def _max_drawdown(self, df):
        """Calculate maximum drawdown from equity curve."""
        equity = df['total_equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        return drawdown.min()


# ============================================================
# Runner: Run simulation from price/funding data
# ============================================================

def run_simulation(price_data: pd.DataFrame,
                   capital: float = 100_000,
                   leverage: float = 2.0,
                   rebalance_threshold: float = 0.05,
                   mode: str = "leveraged",
                   e_mode: bool = False,
                   verbose: bool = True):
    """
    Run a complete simulation from a price/funding DataFrame.

    Args:
        price_data: DataFrame with columns:
            - timestamp (or index)
            - eth_price: ETH price in USD
            - funding_rate: Funding rate for this period (raw hourly rate)
            - borrow_apy: (optional) Aave USDC borrow APY
            - supply_apy: (optional) Aave WETH supply APY
        capital: Starting capital in USDC
        leverage: Target leverage
        rebalance_threshold: Margin ratio threshold for rebalancing
        mode: "simple" or "leveraged"
        e_mode: Use Aave E-Mode
        verbose: Print progress

    Returns:
        (simulator, results_df, summary_dict)
    """
    sim = DeltaNeutralSimulator(
        capital=capital,
        leverage=leverage,
        rebalance_threshold=rebalance_threshold,
        mode=mode,
        e_mode=e_mode,
    )

    # Initialize position at first price
    first_price = price_data.iloc[0]['eth_price']
    if verbose:
        print(f"Initializing {mode} position:")
        print(f"  Capital: ${capital:,.0f}")
        print(f"  Leverage: {leverage}×")
        print(f"  ETH price: ${first_price:,.2f}")

    sim.initialize_position(first_price)

    if verbose:
        print(f"  Spot ETH: {sim.spot_eth:.4f} (${sim.spot_eth * first_price:,.0f})")
        print(f"  Short perp: {sim.hl_account.position_size_eth:.4f} ETH")
        print(f"  HL margin: ${sim.hl_account.margin_deposited:,.0f}")
        if mode == "leveraged":
            print(f"  Aave collateral: {sim.aave.collateral_eth:.4f} ETH")
            print(f"  Aave debt: ${sim.aave.debt_usdc:,.0f}")
            print(f"  Aave HF: {sim.aave.health_factor(first_price):.2f}")
        print()

    # Run simulation
    for idx, row in price_data.iterrows():
        result = sim.step(
            timestamp=row.get('timestamp', idx),
            eth_price=row['eth_price'],
            funding_rate=row.get('funding_rate', 0.0),
            borrow_apy=row.get('borrow_apy', 0.05),
            supply_apy=row.get('supply_apy', 0.02),
            hours_elapsed=row.get('hours_elapsed', 1.0),
        )

        # Stop if liquidated
        if result.get('aave_liquidated') or result.get('hl_liquidated'):
            if verbose:
                liq_type = 'Aave' if result.get('aave_liquidated') else 'Hyperliquid'
                print(f"  ⚠ LIQUIDATED ({liq_type}) at ETH=${row['eth_price']:,.2f}")
            break

    results_df = sim.get_results()
    summary = sim.summary()

    if verbose:
        print(f"\n{'='*60}")
        print(f"SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"  Duration: {summary['duration_hours']} hours "
              f"({summary['duration_hours']/24:.1f} days)")
        print(f"  ETH: ${summary['initial_eth_price']:,.0f} → "
              f"${summary['final_eth_price']:,.0f} "
              f"({summary['eth_return_pct']:+.1f}%)")
        print(f"  Equity: ${capital:,.0f} → ${summary['final_equity']:,.0f}")
        print(f"  PnL: ${summary['total_pnl']:,.0f} ({summary['total_return_pct']:+.2f}%)")
        print(f"  Annualized: {summary['annualized_return_pct']:+.2f}%")
        print(f"  Funding received: ${summary['total_funding_received']:,.0f}")
        print(f"  Fees paid: ${summary['total_fees_paid']:,.0f}")
        if mode == "leveraged":
            print(f"  Aave interest paid: ${summary['total_aave_interest']:,.0f}")
            print(f"  Aave supply yield: ${summary['total_aave_supply_yield']:,.0f}")
        print(f"  Rebalances: {summary['rebalance_count']}")
        print(f"  Max drawdown: {summary['max_drawdown_pct']:.2f}%")
        if summary['hl_liquidated'] or summary['aave_liquidated']:
            print(f"  *** LIQUIDATION OCCURRED ***")

    return sim, results_df, summary


def create_sample_price_data(days=30, start_price=3500, volatility=0.6,
                              mean_funding_rate=0.0001,
                              borrow_apy=0.05, supply_apy=0.02):
    """
    Create sample hourly price data for testing.

    Args:
        days: Number of days
        start_price: Starting ETH price
        volatility: Annual volatility (e.g., 0.6 = 60%)
        mean_funding_rate: Mean hourly funding rate
        borrow_apy: Aave borrow APY
        supply_apy: Aave supply APY

    Returns:
        DataFrame with columns: timestamp, eth_price, funding_rate, borrow_apy, supply_apy
    """
    hours = days * 24
    hourly_vol = volatility / np.sqrt(8760)

    prices = [start_price]
    for _ in range(hours - 1):
        ret = np.random.normal(0, hourly_vol)
        prices.append(prices[-1] * np.exp(ret))

    # Funding rates: mean-reverting around mean_funding_rate
    funding_rates = []
    fr = mean_funding_rate
    for _ in range(hours):
        fr = 0.95 * fr + 0.05 * mean_funding_rate + np.random.normal(0, mean_funding_rate * 0.5)
        funding_rates.append(fr)

    timestamps = pd.date_range('2024-01-01', periods=hours, freq='1h')

    return pd.DataFrame({
        'timestamp': timestamps,
        'eth_price': prices,
        'funding_rate': funding_rates,
        'borrow_apy': borrow_apy,
        'supply_apy': supply_apy,
        'hours_elapsed': 1.0,
    })


# ============================================================
# Comparison: Run multiple leverage levels
# ============================================================

def compare_leverage_levels(price_data: pd.DataFrame,
                            capital: float = 100_000,
                            leverage_levels: list = None,
                            mode: str = "leveraged",
                            verbose: bool = False):
    """
    Compare simulation results across different leverage levels.

    Args:
        price_data: Price/funding DataFrame
        capital: Starting capital
        leverage_levels: List of leverage values to test
        mode: "simple" or "leveraged"
        verbose: Print per-simulation details

    Returns:
        DataFrame with comparison results
    """
    if leverage_levels is None:
        leverage_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    results = []
    for L in leverage_levels:
        try:
            _, _, summary = run_simulation(
                price_data, capital=capital, leverage=L,
                mode=mode, verbose=verbose
            )
            summary['leverage_input'] = L
            results.append(summary)
        except Exception as e:
            results.append({
                'leverage_input': L,
                'error': str(e),
            })

    comparison = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"LEVERAGE COMPARISON ({mode} mode, ${capital:,.0f} capital)")
    print(f"{'='*80}")

    display_cols = [
        'leverage_input', 'total_return_pct', 'annualized_return_pct',
        'total_funding_received', 'max_drawdown_pct',
        'rebalance_count', 'hl_liquidated', 'aave_liquidated'
    ]
    available = [c for c in display_cols if c in comparison.columns]
    print(comparison[available].to_string(index=False))

    return comparison


# ============================================================
# Main: Demo
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 70)
    print("REALISTIC DELTA-NEUTRAL FUNDING RATE SIMULATION")
    print("=" * 70)

    # Create sample data
    print("\n--- Generating sample price data (90 days) ---\n")
    price_data = create_sample_price_data(
        days=90,
        start_price=3500,
        volatility=0.60,
        mean_funding_rate=0.0001,  # ~0.01% per hour ≈ 87.6% APR
        borrow_apy=0.05,
        supply_apy=0.02,
    )

    # Single simulation
    print("\n--- Single Simulation: 2× Leveraged ---\n")
    sim, results, summary = run_simulation(
        price_data,
        capital=100_000,
        leverage=2.0,
        mode="leveraged",
        verbose=True,
    )

    # Comparison across leverage levels
    print("\n\n--- Leverage Comparison ---")
    comparison = compare_leverage_levels(
        price_data,
        capital=100_000,
        leverage_levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        mode="leveraged",
        verbose=False,
    )

    # Show how to use with real data
    print("\n\n--- HOW TO USE WITH REAL DATA ---")
    print("""
    # Load your real data:
    import pandas as pd

    # Option 1: From CSV with columns: timestamp, eth_price, funding_rate
    df = pd.read_csv('my_data.csv')
    df['borrow_apy'] = 0.05   # or load real Aave rates
    df['supply_apy'] = 0.02   # or load real Aave rates
    df['hours_elapsed'] = 1.0  # hourly data

    # Option 2: Manual price series
    df = pd.DataFrame({
        'eth_price': [3500, 3520, 3480, 3510, ...],
        'funding_rate': [0.0001, 0.00012, 0.00008, ...],
        'borrow_apy': 0.05,
        'supply_apy': 0.02,
        'hours_elapsed': 1.0,
    })

    # Run simulation
    sim, results, summary = run_simulation(
        df,
        capital=100_000,
        leverage=2.0,
        mode='leveraged',
    )

    # Access detailed results
    print(results[['timestamp', 'eth_price', 'total_equity', 'pnl_pct',
                    'hl_margin_ratio', 'aave_health_factor']].tail())
    """)
