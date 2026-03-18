"""
Policy Classes and Optimization
================================

Implements:
- Constant leverage policies (baselines B1, B2, B3)
- Adaptive Leverage Governor (ALG)
- Dynamic leverage policy (lookup-table based)
- Policy optimizer with constraints
"""

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm


class ConstantLeveragePolicy:
    """Constant leverage policy (baseline)."""
    
    def __init__(self, L, epsilon, name='Constant'):
        """
        Args:
            L: Fixed leverage
            epsilon: Fixed rebalancing band
            name: Policy name
        """
        self.L = L
        self.epsilon = epsilon
        self.name = name
    
    def get_leverage(self, Q, state):
        """
        Return leverage and epsilon for given state.
        
        Args:
            Q: Vault size (unused for constant policy)
            state: Market state (unused for constant policy)
        
        Returns:
            (L, epsilon) tuple
        """
        return self.L, self.epsilon
    
    def __repr__(self):
        return f"{self.name}(L={self.L}, ε={self.epsilon})"


class DynamicLeveragePolicy:
    """Dynamic leverage policy optimized for each (Q, regime)."""
    
    def __init__(self, policy_table, name='Dynamic'):
        """
        Args:
            policy_table: Dict mapping (Q, regime) -> (L, epsilon)
            name: Policy name
        """
        self.policy_table = policy_table
        self.name = name
        self.Q_grid = sorted(set(q for q, _ in policy_table.keys()))
    
    def get_leverage(self, Q, state):
        """
        Return optimal leverage for given Q and state.
        
        Args:
            Q: Vault size
            state: Market state dict with 'regime' key
        
        Returns:
            (L, epsilon) tuple
        """
        regime = state.get('regime', 'default')
        
        # Find closest Q in grid
        Q_closest = min(self.Q_grid, key=lambda q: abs(q - Q))
        
        # Lookup
        key = (Q_closest, regime)
        if key in self.policy_table:
            return self.policy_table[key]
        
        # Fallback: use default regime
        key_default = (Q_closest, 'default')
        if key_default in self.policy_table:
            return self.policy_table[key_default]
        
        # Last resort: return first available for this Q
        for k, v in self.policy_table.items():
            if k[0] == Q_closest:
                return v
        
        # Ultimate fallback
        return 2.0, 0.05
    
    def __repr__(self):
        return f"{self.name}(Q_grid={len(self.Q_grid)} points, regimes={len(set(r for _, r in self.policy_table.keys()))})"


class AdaptiveLeverageGovernor:
    """
    Adaptive Leverage Governor (ALG): real-time per-epoch optimization.

    At each epoch, given current vault size Q and market state s, the
    governor solves for the carry-maximizing (L, epsilon):

        L*(epsilon) = (f - r_b) * dt * epsilon^2 / (2 * c_reb * sigma_e^2 * K)

    where K = cost-per-rebalance / Q.  L* is the closed-form FOC for
    leverage; epsilon is optimized via a fine grid.  This gives the
    theoretical upper bound on what an adaptive policy can achieve
    (without inter-epoch transition costs).
    """

    def __init__(self, cost_model, L_max=5.0, name='ALG (Adaptive)'):
        self.model = cost_model
        self.name = name
        self.L_min = 1.0
        self.L_max = L_max
        self.epsilon_grid = np.linspace(0.02, 0.15, 14)

        # Cache model params for fast inner loop
        self.c_reb = cost_model.params['c_reb']
        self.beta = cost_model.params['beta']
        self.alpha_perp = cost_model.params['alpha_perp']
        self.phi_perp = cost_model.params['phi_perp']
        # Empirical spot impact params
        self.spot_impact_a = cost_model.params['spot_impact_a']
        self.spot_impact_b = cost_model.params['spot_impact_b']
        self.phi_spot = cost_model.params['phi_spot']
        self.c_fixed = cost_model.params['c_fixed']
        self.rate_curve = cost_model.params['rate_curve']

    def _get_rates(self, state):
        """Extract borrow and supply rates, with rate-curve fallback."""
        if state.get('borrow_rate') is not None and state.get('supply_rate') is not None:
            return state['borrow_rate'], state['supply_rate']
        U = state.get('U', 0.80)
        rc = self.rate_curve
        if U <= rc['U_kink']:
            r_b = rc['r0'] + rc['r1'] * U
        else:
            r_b = rc['r0'] + rc['r1'] * rc['U_kink'] + rc['r2'] * (U - rc['U_kink'])
        r_s = U * r_b * 0.90
        return r_b, r_s

    def get_leverage(self, Q, state):
        """
        Compute optimal (L, epsilon) for current vault size and market state.

        Args:
            Q: Vault size (USD)
            state: Dict with f, sigma, D_spot, D_perp, U/borrow_rate/supply_rate

        Returns:
            (L, epsilon) tuple
        """
        f = state['f']
        sigma = state['sigma']
        D_spot = state['D_spot']
        D_perp = state['D_perp']
        dt = state.get('dt', 8 / 8760)

        r_b, r_s = self._get_rates(state)

        sigma_e_sq = sigma ** 2 * (8 / 8760)  # epoch variance
        marginal_carry = (f - r_b) * dt        # carry per unit L, net of borrow

        best_rnet = -np.inf
        best_L = self.L_min
        best_eps = 0.05

        for eps in self.epsilon_grid:
            # Q/2 capital per side; trade per side = eps * Q/2
            trade = eps * (Q / 2)

            # Cost per rebalance / Q  (=K)
            # Spot: empirical Uniswap model (fee included)
            spot_slip_bps = (self.phi_spot * 10000
                             + self.spot_impact_a * trade ** self.spot_impact_b)
            spot_cost_per_Q = (spot_slip_bps / 10000) * trade / Q
            # Perp: Gatheral model + perp fee
            perp_frac = (trade / D_perp) ** self.beta
            perp_cost_per_Q = (self.phi_perp * eps / 2
                               + self.alpha_perp * (eps / 2) * perp_frac)
            K = spot_cost_per_Q + perp_cost_per_Q + self.c_fixed / Q

            # Analytical L* from first-order condition: d(r_net)/dL = 0
            if marginal_carry > 0 and sigma_e_sq > 0 and K > 0:
                L_star = (marginal_carry * eps ** 2) / (2 * self.c_reb * sigma_e_sq * K)
            else:
                L_star = self.L_min

            L = float(np.clip(L_star, self.L_min, self.L_max))

            # Evaluate annualized net carry rate (per unit Q)
            lambda_reb = self.c_reb * (L ** 2 * sigma_e_sq) / (eps ** 2)
            gross = f * L * dt / 2        # funding on L * Q/2
            reb = lambda_reb * K
            borrow = ((L - 1) * r_b - r_s) * dt / 2 if L > 1 else -r_s * dt / 2

            r_net = gross - reb - borrow

            if r_net > best_rnet:
                best_rnet = r_net
                best_L = L
                best_eps = eps

        return best_L, best_eps

    def __repr__(self):
        return f"{self.name}(L=[{self.L_min}, {self.L_max}])"


class SmoothedAdaptiveLeverageGovernor(AdaptiveLeverageGovernor):
    """
    ALG with leverage-transition costs (ALG-S).

    Extends the myopic ALG by penalizing leverage changes between epochs.
    When the governor changes leverage from L_prev to L_new, the vault
    must trade |L_new - L_prev| * Q/2 notional on each side, incurring
    fees and market impact.  The transition cost (per unit Q) is:

        γ(ΔL, Q) = |ΔL| * (φ + α_spot*(|ΔL|*Q/2 / D_spot)^β
                                 + α_perp*(|ΔL|*Q/2 / D_perp)^β)

    The smoothed governor subtracts this one-time cost from the epoch
    carry when evaluating candidate (L, ε) pairs, producing a policy
    that changes leverage less frequently and by smaller amounts.
    """

    def __init__(self, cost_model, L_max=5.0, name='ALG-S (Smoothed)'):
        super().__init__(cost_model, L_max=L_max, name=name)
        self._prev_L = None

    def reset(self):
        """Reset state between independent simulation runs."""
        self._prev_L = None

    def get_leverage(self, Q, state):
        """
        Compute optimal (L, ε) accounting for the cost of changing
        leverage from the previous epoch.

        On the first call (no previous state), falls back to the
        myopic ALG.
        """
        if self._prev_L is None:
            L, eps = super().get_leverage(Q, state)
            self._prev_L = L
            return L, eps

        f = state['f']
        sigma = state['sigma']
        D_spot = state['D_spot']
        D_perp = state['D_perp']
        dt = state.get('dt', 8 / 8760)
        r_b, r_s = self._get_rates(state)

        sigma_e_sq = sigma ** 2 * (8 / 8760)
        marginal_carry = (f - r_b) * dt

        best_rnet = -np.inf
        best_L = self._prev_L
        best_eps = 0.05

        for eps in self.epsilon_grid:
            trade = eps * (Q / 2)
            # K: same empirical spot + Gatheral perp as base ALG
            spot_slip_bps = (self.phi_spot * 10000
                             + self.spot_impact_a * trade ** self.spot_impact_b)
            spot_cost_per_Q = (spot_slip_bps / 10000) * trade / Q
            perp_frac = (trade / D_perp) ** self.beta
            perp_cost_per_Q = (self.phi_perp * eps / 2
                               + self.alpha_perp * (eps / 2) * perp_frac)
            K = spot_cost_per_Q + perp_cost_per_Q + self.c_fixed / Q

            if marginal_carry > 0 and sigma_e_sq > 0 and K > 0:
                L_star = (marginal_carry * eps ** 2) / (2 * self.c_reb * sigma_e_sq * K)
            else:
                L_star = self.L_min

            L = float(np.clip(L_star, self.L_min, self.L_max))

            # Epoch carry (same as base ALG)
            lambda_reb = self.c_reb * (L ** 2 * sigma_e_sq) / (eps ** 2)
            gross = f * L * dt / 2
            reb = lambda_reb * K
            borrow = ((L - 1) * r_b - r_s) * dt / 2 if L > 1 else -r_s * dt / 2
            r_net = gross - reb - borrow

            # Transition cost: trading |ΔL| * Q/2 on each side
            delta_L = abs(L - self._prev_L)
            if delta_L > 0:
                trade_transition = delta_L * (Q / 2)
                # Spot: empirical model on transition trade
                tc_spot_bps = (self.phi_spot * 10000
                               + self.spot_impact_a * trade_transition ** self.spot_impact_b)
                tc_spot = (tc_spot_bps / 10000) * trade_transition / Q
                # Perp: Gatheral
                tc_perp = (self.phi_perp * delta_L / 2
                           + self.alpha_perp * (delta_L / 2)
                             * (trade_transition / D_perp) ** self.beta)
                r_net -= (tc_spot + tc_perp)

            if r_net > best_rnet:
                best_rnet = r_net
                best_L = L
                best_eps = eps

        self._prev_L = best_L
        return best_L, best_eps

    def __repr__(self):
        return f"{self.name}(L=[{self.L_min}, {self.L_max}], smoothed)"


class PolicyOptimizer:
    """
    Optimize leverage and epsilon subject to constraints.

    For each (Q, regime), find (L, epsilon) that maximizes mean net carry
    while satisfying risk constraints.
    """
    
    def __init__(self, cost_model, constraints=None):
        """
        Args:
            cost_model: VaultCostModel instance
            constraints: Dict with constraint parameters
        """
        self.model = cost_model
        
        # Default constraints
        self.constraints = constraints or {
            'max_cvar_95': 0.05,           # 5% CVaR limit
            'min_mean_carry': 0.0,         # Must be profitable
            'max_rebalance_freq': 5.0,     # Max 5 rebalances per epoch
            'min_L': 1.0,
            'max_L': 5.0,
            'min_epsilon': 0.02,
            'max_epsilon': 0.15,
        }
    
    def optimize_for_state(self, Q, data_regime, L_grid=None, epsilon_grid=None):
        """
        Find optimal (L, epsilon) for given Q and regime data.
        
        Args:
            Q: Vault size
            data_regime: DataFrame with data for this regime
            L_grid: Leverage values to search
            epsilon_grid: Epsilon values to search
        
        Returns:
            Dict with optimal (L, epsilon) and performance metrics
        """
        
        if L_grid is None:
            L_grid = np.linspace(
                self.constraints['min_L'],
                self.constraints['max_L'],
                20
            )
        
        if epsilon_grid is None:
            epsilon_grid = np.linspace(
                self.constraints['min_epsilon'],
                self.constraints['max_epsilon'],
                15
            )
        
        best_policy = None
        best_carry = -np.inf
        
        # Grid search
        for L, epsilon in product(L_grid, epsilon_grid):
            # Simulate performance
            results = self.model.simulate_over_data(Q, L, epsilon, data_regime)
            
            # Calculate metrics
            mean_carry = results['net_carry_annual_pct'].mean()
            cvar_95 = self.calculate_cvar(results['net_carry_annual_pct'], alpha=0.95)
            mean_freq = results['lambda_reb'].mean()
            
            # Check constraints
            if cvar_95 > self.constraints['max_cvar_95']:
                continue
            if mean_carry < self.constraints['min_mean_carry']:
                continue
            if mean_freq > self.constraints['max_rebalance_freq']:
                continue
            
            # Update best
            if mean_carry > best_carry:
                best_carry = mean_carry
                best_policy = {
                    'L': L,
                    'epsilon': epsilon,
                    'mean_carry': mean_carry,
                    'cvar_95': cvar_95,
                    'mean_freq': mean_freq,
                    'sharpe': mean_carry / results['net_carry_annual_pct'].std() if results['net_carry_annual_pct'].std() > 0 else 0,
                }
        
        return best_policy
    
    def calculate_cvar(self, returns, alpha=0.95):
        """
        Calculate Conditional Value at Risk (CVaR).
        
        CVaR_α = mean of worst (1-α) outcomes
        
        Args:
            returns: Array of returns
            alpha: Confidence level (e.g., 0.95)
        
        Returns:
            CVaR value (absolute, not percentage)
        """
        threshold = np.percentile(returns, (1 - alpha) * 100)
        tail_losses = returns[returns <= threshold]
        return abs(tail_losses.mean()) if len(tail_losses) > 0 else 0
    
    def build_policy_table(self, Q_grid, data_df, regime_col='regime'):
        """
        Build complete policy table for all (Q, regime) combinations.
        
        Args:
            Q_grid: List of vault sizes to optimize for
            data_df: Master dataset with regime labels
            regime_col: Column name for regime
        
        Returns:
            Dict mapping (Q, regime) -> (L, epsilon)
        """
        
        print("Building dynamic policy table...")
        print(f"Q grid: {len(Q_grid)} points")
        
        regimes = data_df[regime_col].unique()
        print(f"Regimes: {len(regimes)}")
        
        policy_table = {}
        
        total = len(Q_grid) * len(regimes)
        pbar = tqdm(total=total, desc="Optimizing policies")
        
        for Q in Q_grid:
            for regime in regimes:
                # Get data for this regime
                regime_data = data_df[data_df[regime_col] == regime]
                
                if len(regime_data) < 10:
                    # Not enough data, skip
                    pbar.update(1)
                    continue
                
                # Optimize
                result = self.optimize_for_state(Q, regime_data)
                
                if result is not None:
                    policy_table[(Q, regime)] = (result['L'], result['epsilon'])
                
                pbar.update(1)
        
        pbar.close()
        
        print(f"✅ Built policy table with {len(policy_table)} entries")
        
        return policy_table
    
    def create_dynamic_policy(self, Q_grid, data_df):
        """
        Create DynamicLeveragePolicy from data.
        
        Args:
            Q_grid: List of vault sizes
            data_df: Master dataset
        
        Returns:
            DynamicLeveragePolicy instance
        """
        
        policy_table = self.build_policy_table(Q_grid, data_df)
        return DynamicLeveragePolicy(policy_table, name='Dynamic')


def create_baseline_policies(cost_model=None):
    """
    Create baseline policies for comparison.

    Args:
        cost_model: VaultCostModel instance.  When provided the Adaptive
                    Leverage Governor is included in the returned dict.

    Returns:
        Dict with policy name -> Policy instance
    """

    baselines = {
        'B1_Industry': ConstantLeveragePolicy(
            L=2.0,
            epsilon=0.05,
            name='B1 (Industry Standard)'
        ),

        'B2_Conservative': ConstantLeveragePolicy(
            L=1.2,
            epsilon=0.10,
            name='B2 (Conservative)'
        ),

        'B3_Aggressive': ConstantLeveragePolicy(
            L=2.5,
            epsilon=0.03,
            name='B3 (Aggressive)'
        ),
    }

    if cost_model is not None:
        baselines['ALG_Adaptive'] = AdaptiveLeverageGovernor(
            cost_model, L_max=5.0, name='ALG (Adaptive)'
        )
        baselines['ALGS_Smoothed'] = SmoothedAdaptiveLeverageGovernor(
            cost_model, L_max=5.0, name='ALG-S (Smoothed)'
        )

    return baselines


def evaluate_policy(policy, Q, data_df, cost_model):
    """
    Evaluate policy performance over data.
    
    Args:
        policy: Policy instance
        Q: Vault size
        data_df: Master dataset
        cost_model: VaultCostModel instance
    
    Returns:
        Dict with performance metrics
    """
    
    results_list = []
    
    for idx, row in data_df.iterrows():
        state = {
            'f': row['funding_rate_annual'],
            'sigma': row['realized_vol'],
            'D_spot': row['depth_spot'],
            'D_perp': row['depth_perp'],
            'U': row['utilization'],
            'regime': row.get('regime', 'default'),
        }
        
        # Get policy decision
        L, epsilon = policy.get_leverage(Q, state)
        
        # Calculate net carry
        carry = cost_model.total_net_carry(Q, L, epsilon, state)
        
        results_list.append({
            'timestamp': row['timestamp'],
            'net_carry': carry['net_carry'],
            'net_carry_pct': carry['net_carry_pct'] * 100,
            'L': L,
            'epsilon': epsilon,
        })
    
    results_df = pd.DataFrame(results_list)
    
    # Calculate statistics
    stats = {
        'mean_carry': results_df['net_carry_pct'].mean(),
        'median_carry': results_df['net_carry_pct'].median(),
        'std_carry': results_df['net_carry_pct'].std(),
        'min_carry': results_df['net_carry_pct'].min(),
        'max_carry': results_df['net_carry_pct'].max(),
        'sharpe': results_df['net_carry_pct'].mean() / results_df['net_carry_pct'].std() if results_df['net_carry_pct'].std() > 0 else 0,
        'negative_epochs': (results_df['net_carry_pct'] < 0).sum(),
        'negative_pct': (results_df['net_carry_pct'] < 0).mean() * 100,
    }
    
    return stats, results_df


if __name__ == '__main__':
    """
    Create baseline policies and save to JSON.
    """
    
    import json
    
    print("="*70)
    print("CREATING BASELINE POLICIES")
    print("="*70)
    
    # Create baselines
    baselines = create_baseline_policies()
    
    print("\nBaseline policies created:")
    for name, policy in baselines.items():
        print(f"  {name}: {policy}")
    
    # Save policy definitions
    policy_defs = {}
    for name, policy in baselines.items():
        policy_defs[name] = {
            'L': policy.L,
            'epsilon': policy.epsilon,
            'name': policy.name
        }
    
    with open('./results/baseline_policies.json', 'w') as f:
        json.dump(policy_defs, f, indent=2)
    
    print("\n✅ Policies saved to ./results/baseline_policies.json")