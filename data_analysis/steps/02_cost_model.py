"""
Vault Cost Model
=================

Core model for calculating net carry and all cost components.
Implements the complete friction model from the paper.
"""

import numpy as np
import pandas as pd


class VaultCostModel:
    """
    Complete cost model for funding-rate capture vault.
    
    Calculates:
    - Gross carry (funding revenue)
    - Trading fees
    - Market impact (convex slippage)
    - Rebalancing frequency
    - Borrow costs
    - Total net carry
    """
    
    def __init__(self, params):
        """
        Initialize cost model with parameters.

        Args:
            params: Dict with model parameters
                - phi_spot: Spot trading fee rate (e.g., 0.0005 = 5bps for Uniswap 0.05%)
                - phi_perp: Perp trading fee rate (e.g., 0.00045 = 4.5bps HL base tier)
                - spot_impact_a: Empirical spot impact coefficient (default 0.00464)
                - spot_impact_b: Empirical spot impact exponent (default 0.6778)
                - alpha_perp: Perp impact scale (Gatheral, e.g., 0.0008)
                - beta: Perp impact convexity (Gatheral, e.g., 1.3)
                - c_reb: Rebalancing frequency constant (e.g., 0.733)
                - c_fixed: Fixed cost per rebalance in USD (e.g., 2.0)
                - rate_curve: Dict with {r0, r1, r2, U_kink}

        Spot impact uses empirical model fitted from 1.8M real Uniswap V3
        swaps (USDC/WETH & WETH/USDT, 0.05% pools, Oct 2023 – Dec 2024):

            spot_slippage_bps(size) = phi_spot*10000 + a × size^b

        where a=0.00464, b=0.6778 (P95 fit, R²=0.85). The pool fee
        (phi_spot) is already included in the Uniswap execution price,
        so phi_spot here represents the fee component and the power law
        captures the additional AMM curve slippage.

        Perp impact still uses the Gatheral (2010) model since Hyperliquid
        is an order book, not an AMM.
        """
        self.params = params
        self._validate_params()

    def _validate_params(self):
        """Validate all required parameters present."""
        required = [
            'phi_spot', 'phi_perp',
            'c_reb', 'c_fixed', 'rate_curve'
        ]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        # Defaults for empirical spot impact (from Uniswap data)
        self.params.setdefault('spot_impact_a', 0.00463667)
        self.params.setdefault('spot_impact_b', 0.6778)
        # Defaults for Gatheral perp impact
        self.params.setdefault('alpha_perp', 0.0008)
        self.params.setdefault('beta', 1.3)
        # Legacy: keep alpha_spot for backward compat but not used
        self.params.setdefault('alpha_spot', 0.001)
    
    def gross_carry(self, Q, L, funding_rate, epoch_duration=8/8760):
        """
        Calculate gross carry from funding payments.

        Q is total vault capital, split equally between spot and perp sides.
        Each side has capital Q/2, leveraged at L, giving notional = L * Q/2.
        Funding is earned on the short-perp notional = L * Q/2.

        Args:
            Q: Total vault capital (USD)
            L: Per-side leverage
            funding_rate: Annual funding rate
            epoch_duration: Epoch duration as fraction of year

        Returns:
            Gross carry for this epoch (USD)
        """
        notional = L * (Q / 2)
        carry = funding_rate * notional * epoch_duration
        return carry
    
    def rebalance_frequency(self, L, epsilon, sigma_annual):
        """
        Expected number of rebalances per epoch.

        Uses barrier-crossing approximation:
        λ = c_reb × (L² × σ²_epoch) / ε²

        Args:
            L: Per-side leverage
            epsilon: Rebalancing tolerance band
            sigma_annual: Annual volatility (e.g., 0.60 = 60%)

        Returns:
            Expected rebalances per 8-hour epoch
        """
        # Convert annual volatility to 8-hour epoch
        epoch_hours = 8
        hours_per_year = 8760
        sigma_epoch = sigma_annual * np.sqrt(epoch_hours / hours_per_year)

        # Barrier-crossing frequency
        lambda_reb = (
            self.params['c_reb'] *
            (L**2 * sigma_epoch**2) /
            (epsilon**2)
        )

        return lambda_reb
    
    def fee_cost(self, Q, epsilon):
        """
        Trading fees per rebalance.

        Note: spot fee (phi_spot) is already included in the empirical
        spot slippage model inside impact_cost(). Only the perp fee
        is counted here to avoid double-counting.

        Args:
            Q: Vault size
            epsilon: Rebalancing band (fraction of position to correct)

        Returns:
            Fee cost per rebalance (USD)
        """
        trade_size = epsilon * (Q / 2)

        # Only perp fee — spot fee is inside the empirical impact model
        fee_perp = self.params['phi_perp'] * trade_size

        return fee_perp
    
    def spot_slippage_bps(self, trade_size_usd):
        """
        Empirical spot slippage from real Uniswap V3 data.

        Fitted from 1.8M swaps across USDC/WETH & WETH/USDT 0.05% pools
        (Oct 2023 – Dec 2024). P95 model, R²=0.85.

        The pool fee (phi_spot) is included as the constant term.
        The power law captures additional AMM curve slippage beyond the fee.

        Args:
            trade_size_usd: Trade size in USD

        Returns:
            Slippage in basis points
        """
        fee_bps = self.params['phi_spot'] * 10000  # e.g., 5.0 bps
        a = self.params['spot_impact_a']
        b = self.params['spot_impact_b']
        curve_bps = a * trade_size_usd ** b
        return fee_bps + curve_bps

    def impact_cost(self, Q, epsilon, D_spot, D_perp):
        """
        Market impact (slippage) per rebalance.

        Spot side: empirical model from real Uniswap V3 swap data.
            cost = slippage_bps(trade_size) / 10000 × trade_size

        Perp side: Gatheral (2010) power-law model (Hyperliquid is an
        order book, not an AMM, so empirical AMM model doesn't apply).
            cost = α × trade_size × (trade_size / D_perp)^β

        Args:
            Q: Vault size
            epsilon: Rebalancing band
            D_spot: Spot market depth (USD) — unused in empirical model
            D_perp: Perp market depth (USD)

        Returns:
            Impact cost per rebalance (USD)
        """
        trade_size = epsilon * (Q / 2)

        # Spot impact: empirical Uniswap model
        spot_slip_bps = self.spot_slippage_bps(trade_size)
        impact_spot = (spot_slip_bps / 10000) * trade_size

        # Perp impact: Gatheral model (order book)
        impact_perp = (
            self.params['alpha_perp'] *
            trade_size *
            (trade_size / D_perp) ** self.params['beta']
        )

        return impact_spot + impact_perp
    
    def borrow_cost(self, Q, L, state, epoch_duration=8/8760):
        """
        Net borrow cost (borrow rate - supply rate).

        Uses actual per-asset rates from data when available,
        falls back to rate curve model otherwise.

        Args:
            Q: Vault size
            L: Leverage
            state: Dict with 'borrow_rate' and 'supply_rate' (annual),
                   or 'U' for rate-curve fallback
            epoch_duration: Epoch duration as fraction of year

        Returns:
            Net borrow cost for this epoch (USD)
        """
        # Spot side has Q/2 capital, needs L * Q/2 notional → borrows (L-1) * Q/2
        borrowed = (L - 1) * (Q / 2) if L > 1 else 0
        deposited = Q / 2  # supply rate earned on collateral per side

        if state.get('borrow_rate') is not None and state.get('supply_rate') is not None:
            borrow_rate = state['borrow_rate']
            supply_rate = state['supply_rate']
        else:
            utilization = state.get('U', 0.80)
            rc = self.params['rate_curve']
            if utilization <= rc['U_kink']:
                borrow_rate = rc['r0'] + rc['r1'] * utilization
            else:
                borrow_rate = (
                    rc['r0'] +
                    rc['r1'] * rc['U_kink'] +
                    rc['r2'] * (utilization - rc['U_kink'])
                )
            supply_rate = utilization * borrow_rate * 0.90

        interest_paid = borrowed * borrow_rate * epoch_duration
        interest_earned = deposited * supply_rate * epoch_duration

        return interest_paid - interest_earned
    
    def total_rebalance_cost(self, Q, L, epsilon, state):
        """
        Total rebalancing cost for one epoch.
        
        Args:
            Q: Vault size
            L: Leverage
            epsilon: Rebalancing band
            state: Dict with market state
                - sigma: Annual volatility
                - D_spot: Spot depth
                - D_perp: Perp depth
        
        Returns:
            Dict with breakdown and total
        """
        # Frequency
        lambda_reb = self.rebalance_frequency(L, epsilon, state['sigma'])
        
        # Cost per rebalance
        fee = self.fee_cost(Q, epsilon)
        impact = self.impact_cost(Q, epsilon, state['D_spot'], state['D_perp'])
        fixed = self.params['c_fixed']
        
        cost_per_rebalance = fee + impact + fixed
        
        # Total for epoch
        total = lambda_reb * cost_per_rebalance
        
        return {
            'lambda_reb': lambda_reb,
            'fee_cost': fee,
            'impact_cost': impact,
            'fixed_cost': fixed,
            'cost_per_rebalance': cost_per_rebalance,
            'total_cost': total
        }
    
    def total_net_carry(self, Q, L, epsilon, state):
        """
        Complete net carry calculation for one epoch.
        
        Args:
            Q: Vault size
            L: Leverage
            epsilon: Rebalancing band
            state: Dict with all market state
                - f: Annual funding rate
                - sigma: Annual volatility
                - D_spot: Spot depth
                - D_perp: Perp depth
                - U: Aave utilization
                - dt: Epoch duration (default 8/8760)
        
        Returns:
            Dict with complete breakdown
        """
        epoch_duration = state.get('dt', 8/8760)
        
        # Components
        gross = self.gross_carry(Q, L, state['f'], epoch_duration)
        rebalance = self.total_rebalance_cost(Q, L, epsilon, state)
        borrow = self.borrow_cost(Q, L, state, epoch_duration)
        
        # Net carry
        net = gross - rebalance['total_cost'] - borrow
        
        return {
            'gross_carry': gross,
            'rebalance_cost': rebalance['total_cost'],
            'borrow_cost': borrow,
            'net_carry': net,
            'net_carry_pct': (net / Q) * (8760 / 8) if Q > 0 else 0,  # Annualized %
            
            # Detailed breakdown
            'lambda_reb': rebalance['lambda_reb'],
            'fee_cost': rebalance['fee_cost'],
            'impact_cost': rebalance['impact_cost'],
            'fixed_cost': rebalance['fixed_cost'],
        }
    
    def simulate_over_data(self, Q, L, epsilon, data_df):
        """
        Simulate vault performance over historical data.
        
        Args:
            Q: Vault size (constant)
            L: Leverage (constant for this simulation)
            epsilon: Rebalancing band (constant)
            data_df: DataFrame with columns:
                - timestamp
                - funding_rate_annual
                - realized_vol
                - depth_spot
                - depth_perp
                - utilization
        
        Returns:
            DataFrame with epoch-by-epoch results
        """
        results = []
        
        for idx, row in data_df.iterrows():
            # Build state
            state = {
                'f': row['funding_rate_annual'],
                'sigma': row['realized_vol'],
                'D_spot': row['depth_spot'],
                'D_perp': row['depth_perp'],
                'U': row['utilization'],
                'borrow_rate': row.get('borrow_rate', None),
                'supply_rate': row.get('supply_rate', None),
                'dt': 8/8760
            }
            
            # Calculate net carry
            carry = self.total_net_carry(Q, L, epsilon, state)
            
            # Record
            results.append({
                'timestamp': row['timestamp'],
                'Q': Q,
                'L': L,
                'epsilon': epsilon,
                'gross_carry': carry['gross_carry'],
                'rebalance_cost': carry['rebalance_cost'],
                'borrow_cost': carry['borrow_cost'],
                'net_carry': carry['net_carry'],
                'net_carry_annual_pct': carry['net_carry_pct'] * 100,
                'lambda_reb': carry['lambda_reb'],
            })
        
        return pd.DataFrame(results)
    
    def calculate_statistics(self, results_df):
        """
        Calculate summary statistics from simulation results.
        
        Args:
            results_df: Output from simulate_over_data()
        
        Returns:
            Dict with statistics
        """
        return {
            'mean_carry': results_df['net_carry_annual_pct'].mean(),
            'median_carry': results_df['net_carry_annual_pct'].median(),
            'std_carry': results_df['net_carry_annual_pct'].std(),
            'min_carry': results_df['net_carry_annual_pct'].min(),
            'max_carry': results_df['net_carry_annual_pct'].max(),
            'sharpe': (
                results_df['net_carry_annual_pct'].mean() /
                results_df['net_carry_annual_pct'].std()
                if results_df['net_carry_annual_pct'].std() > 0 else 0
            ),
            'mean_rebalances': results_df['lambda_reb'].mean(),
            'total_epochs': len(results_df),
        }


def test_cost_model():
    """Test the cost model with sample inputs."""
    
    print("Testing VaultCostModel...\n")
    
    # Model parameters:
    # - Spot impact: empirical model from 1.8M real Uniswap V3 swaps
    #   (USDC/WETH & WETH/USDT 0.05% pools, Oct 2023 – Dec 2024)
    # - Perp impact: Gatheral (2010), validated against Hyperliquid L2
    # - c_reb: Monte-Carlo-calibrated barrier-crossing constant
    # - phi_spot: Uniswap 0.05% pool fee (included in empirical model)
    # - phi_perp: Hyperliquid base tier taker fee
    params = {
        'phi_spot': 0.0005,          # 5 bps (Uniswap 0.05% pool fee)
        'phi_perp': 0.00045,         # 4.5 bps (Hyperliquid base tier taker)
        'spot_impact_a': 0.00463667, # Empirical: fitted from real swaps
        'spot_impact_b': 0.6778,     # Empirical: P95 power-law exponent
        'alpha_perp': 0.0008,        # Perp impact scale (Gatheral 2010)
        'beta': 1.3,                 # Perp impact convexity
        'c_reb': 0.746,              # Calibrated via Monte Carlo (03_calibration.py)
        'c_fixed': 2.0,              # $2 per rebalance (Arbitrum gas)
        'rate_curve': {
            'r0': 0.00,
            'r1': 0.04,
            'r2': 0.60,
            'U_kink': 0.80
        }
    }
    
    model = VaultCostModel(params)
    
    # Test case: $10M vault, 2x leverage
    Q = 10_000_000
    L = 2.0
    epsilon = 0.05
    
    state = {
        'f': 0.10,           # 10% annual funding
        'sigma': 0.60,       # 60% volatility
        'D_spot': 5_000_000,
        'D_perp': 5_000_000,
        'U': 0.75,           # 75% utilization
    }
    
    # Calculate
    result = model.total_net_carry(Q, L, epsilon, state)
    
    print("Test Case: $10M vault, 2x leverage")
    print("="*50)
    print(f"Gross carry:        ${result['gross_carry']:>12,.0f}")
    print(f"Rebalance cost:     ${result['rebalance_cost']:>12,.0f}")
    print(f"  - Frequency:      {result['lambda_reb']:>12.3f} per epoch")
    print(f"  - Fees:           ${result['fee_cost']:>12,.0f}")
    print(f"  - Impact:         ${result['impact_cost']:>12,.0f}")
    print(f"  - Fixed:          ${result['fixed_cost']:>12,.0f}")
    print(f"Borrow cost:        ${result['borrow_cost']:>12,.0f}")
    print(f"Net carry:          ${result['net_carry']:>12,.0f}")
    print(f"Net carry (annual): {result['net_carry_pct']:>12.2f}%")
    print()
    
    # Sanity checks
    assert result['gross_carry'] > 0, "Gross carry should be positive"
    assert result['net_carry'] < result['gross_carry'], "Net < gross"
    assert result['lambda_reb'] > 0, "Should rebalance sometimes"
    
    print("✅ All tests passed!")


if __name__ == '__main__':
    test_cost_model()