# Funding Vault Capacity Analysis

Complete implementation package for the paper "The Capacity Frontier of Funding-Rate Capture Vaults: Dynamic Leverage Under Size-Dependent Frictions"

## Overview

This package implements:
- Complete cost model with all frictions (fees, impact, rebalancing, borrow costs)
- Monte Carlo calibration of rebalancing frequency
- Policy optimization framework
- All 5 experiments from the paper
- Automated figure generation
- Placeholder value extraction

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place the following CSV files in `./data/`:

**A. Hyperliquid Funding Rates** (`hyperliquid_funding_rates.csv`)
```
timestamp,asset,funding_rate
2024-01-01 00:00:00,ETH,0.00012
2024-01-01 01:00:00,ETH,0.00015
2024-01-01 02:00:00,ETH,-0.00008
...
```
- `timestamp`: UTC timestamp
- `asset`: ETH or BTC
- `funding_rate`: 1-hour funding rate (decimal, not annualized)

**B. Aave v3 Arbitrum USDC Rates** (`aave_v3_arbitrum_usdc_rates.csv`)
```
timestamp,supply_rate,borrow_rate,utilization,total_liquidity
2024-01-01 00:00:00,0.032,0.048,0.75,800000000
2024-01-01 01:00:00,0.033,0.049,0.76,805000000
...
```
- `timestamp`: UTC timestamp
- `supply_rate`: Annual supply APY (decimal, e.g., 0.032 = 3.2%)
- `borrow_rate`: Annual borrow APY (decimal)
- `utilization`: Pool utilization (0-1)
- `total_liquidity`: Total pool size in USD

**C. Depth Data** (`hyperliquid_depth_data.csv`)
```
timestamp,asset,depth_spot,depth_perp
2024-01-01 00:00:00,ETH,5000000,5000000
2024-01-01 01:00:00,ETH,5200000,5100000
...
```
- `timestamp`: UTC timestamp
- `asset`: ETH or BTC
- `depth_spot`: Spot liquidity within 50bps (USD)
- `depth_perp`: Perp liquidity within 50bps (USD)

### 3. Run the Pipeline

**Full analysis (recommended for paper submission):**
```bash
python run_all.py --mode full
```
- Complete grid (30 Q points)
- Dynamic policy optimization
- All experiments
- Runtime: ~60-90 minutes

**Quick analysis (for testing):**
```bash
python run_all.py --mode quick
```
- Reduced grid (15 Q points)
- Baseline policies only
- Runtime: ~15 minutes

**Test run (minimal):**
```bash
python run_all.py --mode test
```
- Minimal grid (5 Q points)
- Baseline policies only
- Runtime: ~5 minutes

## Pipeline Steps

The pipeline runs automatically through these steps:

### Step 1: Data Preprocessing
- Loads and validates all data sources
- Calculates realized volatility
- Adds regime labels (low/med/high vol, deep/thin liquidity)
- Merges into master dataset

**Output:** `./data/master_dataset.csv`

### Step 2: Parameter Calibration
- Calibrates `c_reb` via Monte Carlo barrier-crossing simulation
- Uses literature values for impact parameters (α, β)
- Fits Aave rate curve (r0, r1, r2, U_kink)

**Output:** `./results/calibrated_params.json`

### Step 3: Policy Creation
- Creates baseline policies (B1 Industry, B2 Conservative, B3 Aggressive)
- Optionally creates dynamic policy via grid search optimization

**Output:** Policy objects

### Step 4: Run Experiments
- **E1:** Net carry vs vault size (Figure 1)
- **E2:** Capacity frontier Q_max(r) (Figure 2)
- **E3:** Optimal policy maps L*(Q), ε*(Q) (Figure 3)
- **E4:** Risk metrics (Table 1)
- **E5:** Regime analysis

**Outputs:**
- `./results/E1_carry_vs_size.csv`
- `./results/E2_capacity_frontier.csv`
- `./results/E3_optimal_policies.csv`
- `./results/E4_risk_metrics.csv`
- `./results/E5_regime_analysis.csv`

### Step 5: Generate Outputs
- Creates all figures (PDF + PNG)
- Creates tables (LaTeX + CSV)
- Extracts placeholder values for paper

**Outputs:**
- `./results/figures/figure1_carry_vs_size.pdf`
- `./results/figures/figure2_capacity_frontier.pdf`
- `./results/figures/figure3_optimal_policies.pdf`
- `./results/tables/table1_risk_metrics.tex`
- `./results/placeholder_values.json`

### Step 6: Summary Report
- Generates summary of key findings
- Lists all output files
- Provides next steps

**Output:** `./results/SUMMARY_REPORT.txt`

## Individual Scripts

You can also run each step independently:

```bash
# Step 1: Preprocess data
python 01_data_preprocessing.py

# Step 2: Calibrate parameters
python 03_calibration.py

# Step 3-4: Run experiments manually
python 05_experiments.py

# Step 5: Generate figures
python 06_visualization.py
```

## Output Files

After running, you'll have:

```
vault_analysis/
├── data/
│   ├── master_dataset.csv          # Processed data
│   ├── hyperliquid_funding_rates.csv  # Your input
│   ├── aave_v3_arbitrum_usdc_rates.csv  # Your input
│   └── hyperliquid_depth_data.csv  # Your input
├── results/
│   ├── calibrated_params.json      # Model parameters
│   ├── placeholder_values.json     # Values for paper
│   ├── E1_carry_vs_size.csv       # Experiment results
│   ├── E2_capacity_frontier.csv
│   ├── E3_optimal_policies.csv
│   ├── E4_risk_metrics.csv
│   ├── E5_regime_analysis.csv
│   ├── SUMMARY_REPORT.txt         # Summary
│   ├── figures/
│   │   ├── figure1_carry_vs_size.pdf
│   │   ├── figure2_capacity_frontier.pdf
│   │   ├── figure3_optimal_policies.pdf
│   │   └── *.png                  # PNG versions
│   └── tables/
│       ├── table1_risk_metrics.tex
│       └── table1_risk_metrics.csv
```

## Filling Placeholders in Paper

After running, use `./results/placeholder_values.json` to fill in the paper:

```json
{
  "X": "15.2",
  "X2": "8.7",
  "X3": "3.5",
  "Z": "150",
  "Z2": "280",
  "Z3": "525",
  ...
}
```

**Find and replace in `funding_vault_optimal.tex`:**
- `[X]` → `15.2`
- `[Z]` → `150`
- etc.

Or use the provided script:
```bash
python fill_placeholders.py
```

## Model Parameters

### Default Calibrated Values

After calibration, typical values are:

```json
{
  "phi_spot": 0.0004,     // 4 bps spot fee
  "phi_perp": 0.0004,     // 4 bps perp fee
  "alpha_spot": 0.001,    // Spot impact scale
  "alpha_perp": 0.0008,   // Perp impact scale
  "beta": 1.3,            // Impact convexity
  "c_reb": 0.0008,        // Rebalancing frequency constant
  "c_fixed": 2.0,         // $2 per rebalance (Arbitrum gas)
  "rate_curve": {
    "r0": 0.00,
    "r1": 0.04,
    "r2": 0.60,
    "U_kink": 0.80
  }
}
```

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Out of Memory
Reduce grid size:
```bash
python run_all.py --mode quick
```

### Calibration Taking Too Long
In `03_calibration.py`, reduce `num_sims`:
```python
c_reb, df = calibrate_c_reb(num_sims=1000)  # Default is 5000
```

### Data Format Issues
Check data with:
```python
import pandas as pd
df = pd.read_csv('./data/hyperliquid_funding_rates.csv')
print(df.head())
print(df.dtypes)
```

Ensure:
- Timestamps are parseable by pandas
- Numeric columns are actually numeric (not strings)
- No NaN values in critical columns

## Extending the Code

### Adding a New Baseline Policy

In `04_policies.py`:
```python
baselines['B4_MyPolicy'] = ConstantLeveragePolicy(
    L=1.8,
    epsilon=0.07,
    name='My Custom Policy'
)
```

### Changing Constraint Parameters

In `04_policies.py`, modify `PolicyOptimizer`:
```python
constraints = {
    'max_cvar_95': 0.03,  # Tighter CVaR limit (3% instead of 5%)
    'min_mean_carry': 0.02,  # Require 2% minimum carry
    ...
}
```

### Adding Custom Experiments

In `05_experiments.py`, add new method to `ExperimentRunner`:
```python
def experiment_E6_my_analysis(self):
    # Your custom analysis
    pass
```

## Citation

If you use this code, please cite:

```bibtex
@article{vault_capacity_2025,
  title={The Capacity Frontier of Funding-Rate Capture Vaults: Dynamic Leverage Under Size-Dependent Frictions},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2025}
}
```

## License

MIT License - see LICENSE file

## Support

For questions or issues:
1. Check this README
2. Review code comments
3. Check SUMMARY_REPORT.txt after running
4. Open an issue on GitHub

## Acknowledgments

- Hyperliquid for on-chain perpetual futures data
- Aave for transparent lending protocol data
- Empirical impact model based on Gatheral (2010)
- Barrier-crossing calibration following Almgren-Chriss (2000)

---

**Ready to run!** 🚀

```bash
python run_all.py --mode full
```