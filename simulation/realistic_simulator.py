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
    - Maintenance margin: derived from max leverage and margin tiers
      (not a universal fixed rate — see margin-tiers docs)
    - Funding: hourly accrual
    - Trading fee: tiered by 14-day rolling volume
      Base tier: 0.045% taker, 0.015% maker
    - Liquidation: staged (partial for >$100k notional), not instant full close
      Backstop only when equity < 2/3 of maintenance

    Health factor (analogous to Aave):
        HF = account_equity / maintenance_margin_requirement
        HF < 1.0 → liquidation triggered

    This makes risk comparable across both legs of the strategy.

    Simplification: we use a single maintenance_margin_rate rather than
    the full tiered schedule. This is acceptable for positions within a
    single margin tier.
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
    # ETH max leverage = 50×, initial margin = 1/50 = 2%, maintenance = 1%
    # But for larger positions, tiers increase this. Using 2% as conservative
    # default for positions in the base tier (<$1M notional).
    maintenance_margin_rate: float = 0.02    # 2% base tier for ETH
    taker_fee_rate: float = 0.00045          # 0.045% base tier
    maker_fee_rate: float = 0.00015          # 0.015% base tier
    liquidation_penalty_rate: float = 0.005  # Simplified; real HL uses staged liquidation

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

    def maintenance_margin_required(self, current_price: float):
        """
        Maintenance margin requirement at current mark price.

        maintenance_margin = notional × maintenance_margin_rate

        In reality this follows Hyperliquid's tiered schedule (larger
        positions require proportionally more margin). We use a single
        rate as a simplification for positions within one tier.
        """
        notional = self.position_size_eth * current_price
        return notional * self.maintenance_margin_rate

    def health_factor(self, current_price: float):
        """
        Hyperliquid health factor (analogous to Aave HF).

            HF = account_equity / maintenance_margin_requirement

        Interpretation:
            HF > 1.0  → position is healthy
            HF = 1.0  → at liquidation boundary
            HF < 1.0  → liquidation triggered

        This is equivalent to margin_ratio / maintenance_margin_rate,
        but framing it as a health factor makes risk directly comparable
        with the Aave side of the strategy.
        """
        self.mark_to_market(current_price)
        mm_req = self.maintenance_margin_required(current_price)
        if mm_req <= 0:
            return float('inf')
        return self.equity / mm_req

    def margin_ratio(self, current_price: float):
        """
        Current margin ratio = equity / position_notional.

        Convenience method. Relationship to health factor:
            margin_ratio = HF × maintenance_margin_rate
        """
        self.mark_to_market(current_price)
        notional = self.position_size_eth * current_price
        if notional <= 0:
            return float('inf')
        return self.equity / notional

    def check_liquidation(self, current_price: float):
        """
        Check if position should be liquidated.

        Uses health_factor < 1.0 as the trigger (equivalent to
        margin_ratio < maintenance_margin_rate).

        Simplification: models liquidation as a single terminal event.
        Real Hyperliquid uses staged liquidation (20% partial for >$100k
        notional, backstop at equity < 2/3 maintenance).

        Returns:
            (is_liquidated, loss_amount)
        """
        if self.position_size_eth <= 0:
            return False, 0.0

        hf = self.health_factor(current_price)
        if hf < 1.0:
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
            'health_factor': self.health_factor(current_price),
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

    Aave v3 parameters for WETH on Arbitrum:
    - Max LTV: 82.5%
    - Liquidation threshold: 85.0%
    - Liquidation penalty: 5%
    - Supply APY: variable (typically 1-3%)
    - E-Mode (ETH correlated): LTV 90%, liq threshold 93%

    Source: Aave governance risk-parameter alignment proposal
    https://governance.aave.com/t/arfc-chaos-labs-risk-parameter-updates-ltv-and-lt-alignment/18997

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

    # Protocol parameters (WETH on Aave V3 Arbitrum)
    max_ltv: float = 0.825               # 82.5% LTV
    liquidation_threshold: float = 0.85   # 85.0%
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
                 mode: str = "leveraged",
                 e_mode: bool = False,
                 hl_margin_threshold: float = 0.08,
                 hl_margin_target: float = 0.15,
                 aave_hf_threshold: float = 1.15,
                 aave_hf_target: float = 1.30):
        """
        Args:
            capital: Starting capital in USDC
            leverage: Target leverage (1.0 = no leverage, 2.0 = 2x, etc.)
            mode: "simple" or "leveraged"
            e_mode: Use Aave E-Mode (higher LTV)
            hl_margin_threshold: Rebalance HL when margin ratio drops below this
            hl_margin_target: Target HL margin ratio after rebalance
            aave_hf_threshold: Rebalance Aave if health factor drops below this
            aave_hf_target: Target Aave HF after repayment/reduction
        """
        self.initial_capital = capital
        self.target_leverage = leverage
        self.hl_margin_threshold = hl_margin_threshold
        self.hl_margin_target = hl_margin_target
        self.aave_hf_threshold = aave_hf_threshold
        self.aave_hf_target = aave_hf_target
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

        Capital C is split into two buckets:
            perp_margin (α × C)  → USDC deposited on Hyperliquid
            aave_capital ((1-α) × C) → buys ETH, supplied to Aave, looped

        After Aave looping at LTV ratio r:
            total_spot_value = aave_capital / (1 - r)
            aave_debt = aave_capital × r / (1 - r)

        Short perp notional = total_spot_value (delta neutral).

        Effective leverage on total capital:
            L_eff = total_spot_value / C = (1 - α) / (1 - r)

        Without Aave (simple mode): r = 0, so L_eff = 1 - α < 1.
        """
        if self.mode == "simple":
            self._init_simple(eth_price)
        else:
            self._init_leveraged(eth_price)

        self.is_initialized = True

    def _init_simple(self, eth_price: float):
        """
        Simple mode: no Aave, no looping.

        Capital C split into spot_budget and perp_margin.
        Effective leverage = spot_budget / C = 1 - α (always < 1).

        The 'leverage' parameter here controls the split:
            α = 1 - leverage  (e.g., leverage=0.5 → 50/50 split)

        Since there's no Aave, leverage > 1.0 is impossible.
        We clamp leverage to (0, 1) and interpret it as the
        fraction of capital that goes to spot.
        """
        C = self.cash_usdc
        P = eth_price

        # Clamp: without Aave, max effective leverage < 1.0
        L = min(self.target_leverage, 0.95)  # Leave at least 5% for margin
        if self.target_leverage >= 1.0:
            print(f"  Warning: Simple mode cannot achieve {self.target_leverage}× leverage.")
            print(f"  Clamped to {L:.2f}× (use 'leveraged' mode for ≥1×)")

        # α = fraction for perp margin
        alpha = 1 - L  # e.g., L=0.5 → α=0.5 (50/50)

        spot_budget = (1 - alpha) * C   # = L × C
        perp_margin = alpha * C         # = (1-L) × C

        # Buy spot ETH
        eth_amount = spot_budget / P
        self.spot_eth = eth_amount
        self.spot_entry_price = P

        # Deposit margin and short same ETH amount → delta neutral
        self.hl_account.deposit_margin(perp_margin)
        self.hl_account.open_short(eth_amount, P)

        self.cash_usdc = 0

    def _init_leveraged(self, eth_price: float):
        """
        Leveraged mode using Aave looping.

        Given capital C and target effective leverage L:

        1. Split C into perp_margin (α×C) and aave_capital ((1-α)×C)
        2. Buy aave_capital / P ETH
        3. Supply to Aave → borrow at LTV r → buy more ETH → loop
        4. After looping:
              total_spot_value = aave_capital / (1-r)
              aave_debt = aave_capital × r / (1-r)
        5. Short total_spot_value / P ETH perp (delta neutral)

        Effective leverage: L_eff = (1-α) / (1-r)

        Constraints:
        - Aave HF = liq_threshold / r ≥ target_hf
          → r ≤ liq_threshold / target_hf
        - r < max_ltv (Aave LTV cap)
        - perp_margin > 0

        Solving for α given L and target Aave HF:
        - r = liq_threshold / target_hf  (use max allowed r from HF)
        - α = 1 - L × (1-r)

        If α < 0: not enough capital for this leverage (need r > max_r).
        If α ≥ 1: leverage ≤ 0 (nonsensical).
        """
        C = self.cash_usdc
        L = self.target_leverage
        P = eth_price

        liq_threshold = self.aave.effective_liq_threshold
        max_ltv = self.aave.effective_ltv

        # Target starting Aave HF (minimum acceptable)
        target_aave_hf = 1.2

        # Max LTV from HF constraint: r ≤ liq_threshold / target_hf
        max_r_from_hf = liq_threshold / target_aave_hf

        # Also must respect Aave's hard LTV cap
        max_r = min(max_r_from_hf, max_ltv)

        if L <= 1.0:
            # No Aave needed for L ≤ 1, but user chose leveraged mode
            # Just use a small r (or 0) to keep it simple
            r = 0.0
            alpha = 1 - L  # Same as simple mode
            if alpha < 0.05:
                alpha = 0.05  # Keep minimum margin
        else:
            # Use the max allowed r (tightest Aave HF we accept at start)
            r = max_r

            # α = 1 - L × (1-r)
            alpha = 1 - L * (1 - r)

            if alpha < 0.02:
                # Not enough capital for perp margin at this leverage
                # Reduce leverage to what's achievable
                alpha = 0.02  # Keep 2% minimum for margin
                L = (1 - alpha) / (1 - r)
                print(f"  Warning: Adjusted to {L:.2f}× leverage "
                      f"(insufficient margin for target)")

            if alpha >= 1.0:
                raise ValueError(
                    f"Cannot achieve {L}× leverage. "
                    f"Max with HF≥{target_aave_hf}: "
                    f"{(1-0.02)/(1-max_r):.1f}×"
                )

        # Compute final position sizes
        perp_margin = alpha * C
        aave_capital = (1 - alpha) * C

        if r > 0:
            total_spot_value = aave_capital / (1 - r)
            aave_debt = total_spot_value - aave_capital
        else:
            total_spot_value = aave_capital
            aave_debt = 0

        total_eth = total_spot_value / P
        effective_leverage = total_spot_value / C

        # Execute: supply ETH to Aave
        self.spot_eth = total_eth
        self.spot_entry_price = P
        self.aave.supply_eth(total_eth, P)

        # Borrow from Aave
        if aave_debt > 0:
            self.aave.borrow_usdc(aave_debt, P)

        # Deposit margin and short perp (same ETH quantity → delta neutral)
        self.hl_account.deposit_margin(perp_margin)
        self.hl_account.open_short(total_eth, P)

        self.cash_usdc = 0

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

        BUT: the risk metrics change with price:
        - If ETH goes up: short loses money, HL margin ratio drops, Aave HF rises
        - If ETH goes down: short gains, HL margin ratio rises, Aave HF drops

        We rebalance when HL margin ratio or Aave health factor gets too low.
        """
        rebalanced = False

        # Check Hyperliquid margin ratio
        if self.hl_account.position_size_eth > 0:
            hl_margin_ratio = self.hl_account.margin_ratio(eth_price)

            if hl_margin_ratio < self.hl_margin_threshold:
                self._rebalance_perp_margin(eth_price)
                rebalanced = True

        # Check Aave health factor
        if self.mode == "leveraged" and self.aave.debt_usdc > 0:
            aave_hf = self.aave.health_factor(eth_price)

            if aave_hf < self.aave_hf_threshold:
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
        current_notional = self.hl_account.position_size_eth * eth_price
        needed_equity = self.hl_margin_target * current_notional
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
        target_hf = self.aave_hf_target
        # debt_target = collateral * liq_threshold / target_hf
        debt_target = (
            self.aave.collateral_eth * eth_price
            * self.aave.effective_liq_threshold / target_hf
        )
        repay_amount = self.aave.debt_usdc - debt_target

        if repay_amount <= 0:
            return

        # Source from Hyperliquid margin.
        # Keep enough equity to stay at or above the configured HL margin buffer.
        hl_notional = self.hl_account.position_size_eth * eth_price
        min_hl_equity = self.hl_margin_threshold * hl_notional
        excess_margin = self.hl_account.equity - min_hl_equity
        usdc_available = max(0, excess_margin)
        repay = min(repay_amount, usdc_available)

        if repay > 0:
            self.hl_account.margin_deposited -= repay
            self.aave.repay_usdc(repay)

        # If still not enough, repay debt first then withdraw collateral
        new_hf = self.aave.health_factor(eth_price)
        if new_hf < self.aave_hf_threshold and self.aave.debt_usdc > 0:
            # Close 20% of perp short to free up USDC
            reduce_eth = self.hl_account.position_size_eth * 0.2
            pnl = self.hl_account.close_short(reduce_eth, eth_price)

            # Use freed margin to repay Aave debt first
            remaining_notional = self.hl_account.position_size_eth * eth_price
            min_hl_eq = self.hl_margin_threshold * remaining_notional
            freed_margin = max(0, self.hl_account.equity - min_hl_eq)
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
            'hl_health_factor': self.hl_account.health_factor(eth_price),
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
            'min_hl_health_factor': df['hl_health_factor'].min(),
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
                   mode: str = "leveraged",
                   e_mode: bool = False,
                   hl_margin_threshold: float = 0.08,
                   hl_margin_target: float = 0.15,
                   aave_hf_threshold: float = 1.15,
                   aave_hf_target: float = 1.30,
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
        mode: "simple" or "leveraged"
        e_mode: Use Aave E-Mode
        hl_margin_threshold: Rebalance HL when margin ratio drops below this
        hl_margin_target: Target HL margin ratio after rebalance
        aave_hf_threshold: Rebalance Aave if health factor drops below this
        aave_hf_target: Target Aave HF after repayment/reduction
        verbose: Print progress

    Returns:
        (simulator, results_df, summary_dict)
    """
    sim = DeltaNeutralSimulator(
        capital=capital,
        leverage=leverage,
        mode=mode,
        e_mode=e_mode,
        hl_margin_threshold=hl_margin_threshold,
        hl_margin_target=hl_margin_target,
        aave_hf_threshold=aave_hf_threshold,
        aave_hf_target=aave_hf_target,
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
        notional = sim.spot_eth * first_price
        eff_lev = notional / capital
        print(f"  Spot ETH: {sim.spot_eth:.4f} (${notional:,.0f})")
        print(f"  Short perp: {sim.hl_account.position_size_eth:.4f} ETH")
        print(f"  Effective leverage: {eff_lev:.2f}×")
        print(f"  HL margin: ${sim.hl_account.margin_deposited:,.0f} "
              f"(margin ratio: {sim.hl_account.margin_ratio(first_price):.1%})")
        print(f"  HL HF: {sim.hl_account.health_factor(first_price):.2f}")
        if mode == "leveraged" and sim.aave.debt_usdc > 0:
            print(f"  Aave collateral: {sim.aave.collateral_eth:.4f} ETH "
                  f"(${sim.aave.collateral_eth * first_price:,.0f})")
            print(f"  Aave debt: ${sim.aave.debt_usdc:,.0f} "
                  f"(LTV: {sim.aave.current_ltv(first_price):.1%})")
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
        print(f"  Min HL HF: {summary['min_hl_health_factor']:.2f}")
        if mode == "leveraged" and summary['min_aave_hf'] is not None:
            print(f"  Min Aave HF: {summary['min_aave_hf']:.2f}")
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
        'min_hl_health_factor', 'min_aave_hf',
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
                    'hl_health_factor', 'aave_health_factor']].tail())
    """)
