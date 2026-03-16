# The Adaptive Leverage Governor

**Expanding the Capacity Frontier of Funding-Rate Capture Vaults**

> **[Read the Paper (PDF)](paper/paper.pdf)** | **[View on GitHub](https://github.com/yuchengzhang966/funding-rate-efficient-frontier)**

## Paper

This repository accompanies the paper *"The Adaptive Leverage Governor: Expanding the Capacity Frontier of Funding-Rate Capture Vaults"* by Yucheng Zhang (February 10, 2025).

The paper introduces ALG — a dynamic leverage policy that adapts position sizing to vault scale, achieving significantly higher carry than fixed-leverage baselines while maintaining bounded risk. Key findings:

- **ALG achieves ~35% carry at $1M** vs ~14% for the industry-standard 2x vault
- **Sustains 15%+ carry at $500M** — the only policy to do so at scale
- **Reduces CVaR₉₅ by ~62% and max drawdown by ~71%** compared to B1 at $100M

| | B1 (Industry 2x) | B2 (Conservative) | B3 (Aggressive) | **ALG** |
|---|---|---|---|---|
| Carry @ $1M | 14.4% | 9.9% | 16.4% | **34.8%** |
| Carry @ $500M | 4.7% | 5.6% | 3.6% | **16.6%** |
| CVaR₉₅ @ $100M | 16.7% | 8.2% | 23.6% | **6.3%** |

## Repository Structure

```
├── paper/
│   ├── paper.tex              # LaTeX source
│   ├── paper.pdf              # Compiled paper
│   ├── references.bib         # Bibliography
│   └── figures/               # Figures used in paper
├── data_analysis/
│   ├── steps/
│   │   ├── 01_data_preprocessing.py
│   │   ├── 02_cost_model.py
│   │   ├── 03_calibration.py
│   │   ├── 04_policies.py
│   │   ├── 05_experiments.py
│   │   ├── 06_visualization.py
│   │   ├── data/              # Processed datasets
│   │   └── results/           # Experiment outputs, figures, tables
│   ├── fetch_*.py             # Data collection scripts
│   ├── funding_rate_raw/      # Raw funding rate data
│   ├── borrow_rate_raw/       # Raw borrow rate data
│   └── liquidity_raw/         # Raw order book data
├── run_all.py                 # Full pipeline runner
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Full analysis (recommended)
python run_all.py --mode full

# Quick test
python run_all.py --mode quick
```

### 3. Compile the Paper

```bash
cd paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Pipeline Steps

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `01_data_preprocessing.py` | Load, validate, merge data sources | `data/master_dataset.csv` |
| 2 | `02_cost_model.py` | Cost model with all frictions | — |
| 3 | `03_calibration.py` | Monte Carlo calibration of c_reb | `results/calibrated_params.json` |
| 4 | `04_policies.py` | Baseline + dynamic policy creation | — |
| 5 | `05_experiments.py` | Run experiments E1–E5 | `results/E*.csv` |
| 6 | `06_visualization.py` | Generate figures + tables | `results/figures/`, `results/tables/` |

## Experiments

- **E1:** Net carry vs vault size (Figure 1)
- **E2:** Capacity frontier Q_max(r) (Figure 2)
- **E3:** Optimal policy maps L*(Q), ε*(Q) (Figure 3)
- **E4:** Risk metrics (CVaR₉₅, Sharpe, max drawdown)
- **E5:** Regime analysis (volatility and liquidity regimes)

## Data Sources

- **Hyperliquid** — perpetual futures funding rates and order book depth
- **Aave v3 (Arbitrum)** — USDC/WETH/WBTC borrow rates
- **Binance** — BTC/ETH funding rates (cross-validation)

## Citation

```bibtex
@article{zhang2025alg,
  title={The Adaptive Leverage Governor: Expanding the Capacity Frontier of Funding-Rate Capture Vaults},
  author={Zhang, Yucheng},
  year={2025}
}
```

## License

MIT License
