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
                - phi_spot: Spot trading fee rate (e.g., 0.0004)
                - phi_perp: Perp trading fee rate (e.g., 0.0004)
                - alpha_spot: Spot impact scale (e.g., 0.001)
                - alpha_perp: Perp impact scale (e.g., 0.0008)
                - beta: Impact convexity (e.g., 1.3)
                - c_reb: Rebalancing frequency constant (e.g., 0.0008)
                - c_fixed: Fixed cost per rebalance (e.g., 2.0)
                - rate_curve: Dict with {r0, r1, r2, U_kink}
        """
        self.params = params
        self._validate_params()
    
    def _validate_params(self):
        """Validate all required parameters present."""
        required = [
            'phi_spot', 'phi_perp', 'alpha_spot', 'alpha_perp',
            'beta', 'c_reb', 'c_fixed', 'rate_curve'
        ]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
    
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

        Args:
            Q: Vault size
            epsilon: Rebalancing band (fraction of position to correct)

        Returns:
            Fee cost per rebalance (USD)
        """
        # Each side has capital Q/2; rebalance trade per side = epsilon * Q/2
        trade_size = epsilon * (Q / 2)

        fee_spot = self.params['phi_spot'] * trade_size
        fee_perp = self.params['phi_perp'] * trade_size

        return fee_spot + fee_perp
    
    def impact_cost(self, Q, epsilon, D_spot, D_perp):
        """
        Market impact (slippage) per rebalance.

        Power-law model: fractional impact = α × (size / D)^β
        Dollar impact = size × α × (size / D)^β

        Args:
            Q: Vault size
            L: Leverage
            epsilon: Rebalancing band
            D_spot: Spot market depth (USD)
            D_perp: Perp market depth (USD)

        Returns:
            Impact cost per rebalance (USD)
        """
        trade_size = epsilon * (Q / 2)

        # Spot impact
        impact_spot = (
            self.params['alpha_spot'] *
            trade_size *
            (trade_size / D_spot) ** self.params['beta']
        )

        # Perp impact
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
    
    # Model parameters — must match calibrated_params.json produced by
    # 03_calibration.py.  c_reb = 0.746 is the Monte-Carlo-calibrated
    # barrier-crossing constant; impact parameters are literature values
    # from Gatheral (2010), validated against Hyperliquid L2 snapshots.
    params = {
        'phi_spot': 0.0004,      # 4 bps
        'phi_perp': 0.0004,      # 4 bps
        'alpha_spot': 0.001,     # Impact scale (Gatheral 2010)
        'alpha_perp': 0.0008,    # Impact scale (Gatheral 2010)
        'beta': 1.3,             # Convexity (crypto β ≈ 1.2–1.5)
        'c_reb': 0.746,          # Calibrated via Monte Carlo (03_calibration.py)
        'c_fixed': 2.0,          # $2 per rebalance (Arbitrum gas)
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