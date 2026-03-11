"""
Experiments Runner
==================

Implements all 5 experiments from the paper:
E1: Net carry vs vault size
E2: Capacity frontier
E3: Optimal policy maps
E4: Risk metrics
E5: Regime analysis
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentRunner:
    """Run all experiments and generate results."""
    
    def __init__(self, cost_model, data_df, policies):
        """
        Args:
            cost_model: VaultCostModel instance
            data_df: Master dataset
            policies: Dict of policy_name -> Policy instance
        """
        self.model = cost_model
        self.data = data_df
        self.policies = policies

    @staticmethod
    def _reset_policy(policy):
        """Reset internal state of stateful policies (e.g. SmoothedALG)."""
        if hasattr(policy, 'reset'):
            policy.reset()
    
    def experiment_E1_carry_vs_size(self, Q_grid=None):
        """
        E1: Net carry vs vault size.
        
        For each baseline, calculate mean net carry across vault sizes.
        
        Returns:
            DataFrame with results
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT E1: NET CARRY VS VAULT SIZE")
        print("="*70)
        
        if Q_grid is None:
            Q_grid = np.logspace(6, 8.7, 30)  # $1M to $500M
        
        results = []
        
        for policy_name, policy in self.policies.items():
            print(f"\nRunning {policy_name}...")
            
            for Q in tqdm(Q_grid, desc=f"  Vault sizes"):
                self._reset_policy(policy)
                perf_list = []
                
                for idx, row in self.data.iterrows():
                    state = {
                        'f': row['funding_rate_annual'],
                        'sigma': row['realized_vol'],
                        'D_spot': row['depth_spot'],
                        'D_perp': row['depth_perp'],
                        'U': row['utilization'],
                        'borrow_rate': row.get('borrow_rate', None),
                        'supply_rate': row.get('supply_rate', None),
                        'regime': row.get('regime', 'default'),
                        'dt': 8/8760,
                    }

                    L, epsilon = policy.get_leverage(Q, state)
                    carry = self.model.total_net_carry(Q, L, epsilon, state)
                    perf_list.append(carry['net_carry_pct'] * 100)

                mean_carry = np.mean(perf_list)
                
                results.append({
                    'policy': policy_name,
                    'Q': Q,
                    'mean_carry_annual_pct': mean_carry,
                })
        
        df = pd.DataFrame(results)
        df.to_csv('./results/E1_carry_vs_size.csv', index=False)
        print(f"\n  E1 complete. {len(df)} rows")
        return df
    
    def experiment_E2_capacity_frontier(self, Q_grid=None, r_targets=None):
        """
        E2: Capacity frontier Q_max(r).
        
        For each target carry rate r, find maximum Q that achieves it.
        
        Returns:
            DataFrame with results
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT E2: CAPACITY FRONTIER")
        print("="*70)
        
        if Q_grid is None:
            Q_grid = np.logspace(6, 8.7, 200)
        
        if r_targets is None:
            r_targets = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]  # 2% to 20%
        
        # First, get mean carry for each (policy, Q)
        print("\nCalculating mean carry for all (policy, Q) pairs...")
        
        carry_map = {}
        
        for policy_name, policy in self.policies.items():
            print(f"\n{policy_name}:")
            
            for Q in tqdm(Q_grid):
                self._reset_policy(policy)
                perf_list = []
                
                for idx, row in self.data.iterrows():
                    state = {
                        'f': row['funding_rate_annual'],
                        'sigma': row['realized_vol'],
                        'D_spot': row['depth_spot'],
                        'D_perp': row['depth_perp'],
                        'U': row['utilization'],
                        'borrow_rate': row.get('borrow_rate', None),
                        'supply_rate': row.get('supply_rate', None),
                        'regime': row.get('regime', 'default'),
                    }
                    
                    L, epsilon = policy.get_leverage(Q, state)
                    carry = self.model.total_net_carry(Q, L, epsilon, state)
                    perf_list.append(carry['net_carry_pct'])
                
                carry_map[(policy_name, Q)] = np.mean(perf_list)
        
        # Find Q_max for each (policy, r_target)
        print("\nFinding capacity frontiers...")
        
        results = []
        
        for policy_name in self.policies.keys():
            for r_target in r_targets:
                # Find max Q where mean_carry >= r_target
                Q_max = None
                
                for Q in sorted(Q_grid):
                    if carry_map.get((policy_name, Q), -1) >= r_target:
                        Q_max = Q
                
                results.append({
                    'policy': policy_name,
                    'r_target': r_target * 100,  # Convert to percentage
                    'Q_max': Q_max if Q_max else 0,
                })
        
        df = pd.DataFrame(results)
        df.to_csv('./results/E2_capacity_frontier.csv', index=False)
        print(f"\n  E2 complete. {len(df)} rows")
        return df
    
    def experiment_E3_optimal_policies(self, Q_grid=None):
        """
        E3: Optimal policy maps L*(Q) and ε*(Q).
        
        Show how optimal leverage changes with vault size.
        Only applicable to dynamic policy.
        
        Returns:
            DataFrame with results
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT E3: OPTIMAL POLICY MAPS")
        print("="*70)
        
        if Q_grid is None:
            Q_grid = np.logspace(6, 8.7, 30)
        
        results = []
        
        # For each policy, record (L, epsilon) at each Q
        for policy_name, policy in self.policies.items():
            print(f"\n{policy_name}:")
            
            for Q in tqdm(Q_grid):
                # Get representative state
                sample_state = {
                    'f': self.data['funding_rate_annual'].median(),
                    'sigma': self.data['realized_vol'].median(),
                    'D_spot': self.data['depth_spot'].median(),
                    'D_perp': self.data['depth_perp'].median(),
                    'U': self.data['utilization'].median(),
                    'borrow_rate': self.data['borrow_rate'].median() if 'borrow_rate' in self.data.columns else None,
                    'supply_rate': self.data['supply_rate'].median() if 'supply_rate' in self.data.columns else None,
                    'regime': 'default',
                    'dt': 8/8760,
                }
                
                L, epsilon = policy.get_leverage(Q, sample_state)
                
                results.append({
                    'policy': policy_name,
                    'Q': Q,
                    'L': L,
                    'epsilon': epsilon,
                })
        
        df = pd.DataFrame(results)
        df.to_csv('./results/E3_optimal_policies.csv', index=False)
        print(f"\n  E3 complete. {len(df)} rows")
        return df
    
    def experiment_E4_risk_metrics(self, Q_test=None):
        """
        E4: Risk metrics at selected vault sizes.
        
        Calculate CVaR, max drawdown, Sharpe at specific Q values.
        
        Returns:
            DataFrame with results
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT E4: RISK METRICS")
        print("="*70)
        
        if Q_test is None:
            Q_test = [10_000_000, 50_000_000, 100_000_000, 200_000_000]
        
        results = []
        
        for policy_name, policy in self.policies.items():
            print(f"\n{policy_name}:")
            
            for Q in Q_test:
                print(f"  Q = ${Q:,.0f}")
                self._reset_policy(policy)
                perf_list = []
                
                for idx, row in self.data.iterrows():
                    state = {
                        'f': row['funding_rate_annual'],
                        'sigma': row['realized_vol'],
                        'D_spot': row['depth_spot'],
                        'D_perp': row['depth_perp'],
                        'U': row['utilization'],
                        'borrow_rate': row.get('borrow_rate', None),
                        'supply_rate': row.get('supply_rate', None),
                        'regime': row.get('regime', 'default'),
                    }
                    
                    L, epsilon = policy.get_leverage(Q, state)
                    carry = self.model.total_net_carry(Q, L, epsilon, state)
                    perf_list.append(carry['net_carry_pct'] * 100)
                
                returns = np.array(perf_list)
                
                # Calculate metrics
                mean_carry = returns.mean()
                std_carry = returns.std()
                sharpe = mean_carry / std_carry if std_carry > 0 else 0
                
                # CVaR
                cvar_95 = self.calculate_cvar(returns, alpha=0.95)
                
                # Max drawdown
                cumulative = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = running_max - cumulative
                max_drawdown = drawdown.max()
                
                results.append({
                    'policy': policy_name,
                    'Q': Q,
                    'mean_carry': mean_carry,
                    'std_carry': std_carry,
                    'sharpe': sharpe,
                    'cvar_95': cvar_95,
                    'max_drawdown': max_drawdown,
                })
        
        df = pd.DataFrame(results)
        df.to_csv('./results/E4_risk_metrics.csv', index=False)
        print(f"\n  E4 complete. {len(df)} rows")
        return df
    
    def calculate_cvar(self, returns, alpha=0.95):
        """Calculate CVaR (Conditional Value at Risk)."""
        threshold = np.percentile(returns, (1 - alpha) * 100)
        tail_losses = returns[returns <= threshold]
        return abs(tail_losses.mean()) if len(tail_losses) > 0 else 0
    
    def experiment_E5_regime_analysis(self, Q_test=50_000_000):
        """
        E5: Performance across volatility regimes.
        
        Compare performance in low/med/high vol periods.
        
        Returns:
            DataFrame with results
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT E5: REGIME ANALYSIS")
        print("="*70)
        
        # Group by volatility regime
        regimes = self.data['vol_regime'].unique()
        
        results = []
        
        for policy_name, policy in self.policies.items():
            print(f"\n{policy_name}:")
            
            for regime in regimes:
                regime_data = self.data[self.data['vol_regime'] == regime]
                
                if len(regime_data) == 0:
                    continue
                
                print(f"  {regime}: {len(regime_data)} observations")
                self._reset_policy(policy)
                perf_list = []
                
                for idx, row in regime_data.iterrows():
                    state = {
                        'f': row['funding_rate_annual'],
                        'sigma': row['realized_vol'],
                        'D_spot': row['depth_spot'],
                        'D_perp': row['depth_perp'],
                        'U': row['utilization'],
                        'borrow_rate': row.get('borrow_rate', None),
                        'supply_rate': row.get('supply_rate', None),
                        'regime': row.get('regime', 'default'),
                    }
                    
                    L, epsilon = policy.get_leverage(Q_test, state)
                    carry = self.model.total_net_carry(Q_test, L, epsilon, state)
                    perf_list.append(carry['net_carry_pct'] * 100)
                
                returns = np.array(perf_list)
                
                results.append({
                    'policy': policy_name,
                    'regime': regime,
                    'mean_carry': returns.mean(),
                    'std_carry': returns.std(),
                    'min_carry': returns.min(),
                    'max_carry': returns.max(),
                })
        
        df = pd.DataFrame(results)
        df.to_csv('./results/E5_regime_analysis.csv', index=False)
        
        print(f"\n✅ E5 complete. Saved to ./results/E5_regime_analysis.csv")
        
        return df
    
    def run_all_experiments(self, asset_label=None):
        """
        Run all experiments in sequence.

        Args:
            asset_label: If set, results are saved with this suffix (e.g. 'eth')

        Returns:
            Dict with all results
        """
        suffix = f"_{asset_label}" if asset_label else ""

        print("\n" + "="*70)
        print(f"RUNNING ALL EXPERIMENTS{f' ({asset_label.upper()})' if asset_label else ''}")
        print("="*70)

        results = {}

        results['E1'] = self.experiment_E1_carry_vs_size()
        results['E2'] = self.experiment_E2_capacity_frontier()
        results['E3'] = self.experiment_E3_optimal_policies()
        results['E4'] = self.experiment_E4_risk_metrics()
        results['E5'] = self.experiment_E5_regime_analysis()

        # Save with asset suffix
        for key, df in results.items():
            if asset_label:
                df['asset'] = asset_label.upper()
            df.to_csv(f'./results/{key}{suffix}.csv', index=False)

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*70)
        print(f"\nResults saved to ./results/*{suffix}.csv")

        return results


if __name__ == '__main__':
    """
    Run all experiments per-asset using baseline policies.
    """

    import json
    import os
    import sys
    import importlib.util

    # Import cost model
    spec = importlib.util.spec_from_file_location("cost_model", "./02_cost_model.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    VaultCostModel = cm.VaultCostModel

    # Import policies
    spec = importlib.util.spec_from_file_location("policies", "./04_policies.py")
    pol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pol)
    create_baseline_policies = pol.create_baseline_policies

    print("="*70)
    print("RUNNING PER-ASSET EXPERIMENTS")
    print("="*70)

    # Load data
    print("\nLoading master dataset...")
    data_df = pd.read_csv('./data/master_dataset.csv')
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    print(f"  Loaded {len(data_df)} observations")
    print(f"  Assets: {sorted(data_df['asset'].unique())}")

    # Load params
    print("\nLoading calibrated parameters...")
    with open('./results/calibrated_params.json', 'r') as f:
        params = json.load(f)
    print("  Parameters loaded")

    os.makedirs('./results', exist_ok=True)

    # Run per-asset
    results_by_asset = {}

    for asset in sorted(data_df['asset'].unique()):
        asset_data = data_df[data_df['asset'] == asset].copy()

        print(f"\n{'='*70}")
        print(f"ANALYZING {asset}")
        print(f"{'='*70}")
        print(f"  Observations : {len(asset_data)}")
        print(f"  Date range   : {asset_data['timestamp'].min()} to {asset_data['timestamp'].max()}")
        print(f"  Mean funding : {asset_data['funding_rate_annual'].mean()*100:.2f}%")
        print(f"  Mean vol     : {asset_data['realized_vol'].mean()*100:.1f}%")

        model = VaultCostModel(params)
        policies = create_baseline_policies(cost_model=model)

        print(f"\n  Policies: {list(policies.keys())}")

        runner = ExperimentRunner(model, asset_data, policies)
        results_by_asset[asset] = runner.run_all_experiments(asset_label=asset.lower())

    # Combined comparison files
    if len(results_by_asset) >= 2:
        print(f"\n{'='*70}")
        print("CREATING CROSS-ASSET COMPARISON")
        print(f"{'='*70}")

        for exp in ['E1', 'E2', 'E3', 'E4', 'E5']:
            frames = []
            for asset, res in results_by_asset.items():
                if exp in res:
                    frames.append(res[exp])
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined.to_csv(f'./results/{exp}_comparison.csv', index=False)
                print(f"  Wrote ./results/{exp}_comparison.csv")

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults saved to ./results/")
    print("  Per-asset:  E1_eth.csv, E1_btc.csv, ...")
    print("  Comparison: E1_comparison.csv, ...")

    sys.exit(0)