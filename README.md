# G-STA: Graph Spatio-Temporal Alignment for Trajectories of Directed, Partially Observed Attributed Networks

This repository contains the official implementation and reproducibility package for the G-STA method, including the core alignment framework and subsequent sensitivity and policy analysis.

## 📌 Overview

G-STA is designed for the alignment of trajectories in directed, partially observed attributed networks. This package is divided into two main components:

1. **Core Reproducibility Package**: The fundamental implementation of the G-STA method, providing the tools for trajectory alignment, verification, and empirical analysis.
2. **Sensitivity & Policy Analysis**: An extension focusing on the sensitivity of anomaly scores, regional G-STA, lead-lag gradients, and counterfactual policy experiments.

---

## 📂 Repository Structure

### `core/` (Reproducibility Package v1.1.0)
- **`gsta/`**: The core method implementation.
  - `ufgw.py`: Directed, masked, unbalanced entropic UFGW (Algorithm 1).
  - `softdtw.py`: Soft-DTW value/gradient and lead-lag index.
  - `core.py`: Trajectory-level quantity calculations.
  - `data.py`: UNCTADstat panel construction and preprocessing.
  - `sim.py`: Simulation designs and baselines.
- **`scripts/`**: Automation scripts for reproducing results (e.g., `run_all.sh`).
- **`data/` & `data_download/`**: UNCTADstat data extracts and retrieval scripts.
- **`tests/`**: Verification tests for identity and convergence.
- **`figures/` & `tables/`**: Scripts and files to generate the paper's visualizations and LaTeX tables.
- **`results/`**: All numerical results used in the core paper.

### `sensitivity/` (Policy & Sensitivity Analysis)
- **`code/`**: Scripts for policy experiments, including:
  - `policy_experiment.py`: Experiments for base couplings, maps, regional analysis, and counterfactuals.
  - `verify_sensitivity.py`: Synthetic checks for sensitivity.
  - `make_outputs.py`: Generation of sensitivity-specific figures and tables.
- **`figures/`**: PDF figures for the sensitivity analysis sections.
- **`results/`**: CSV files containing the data behind the sensitivity tables.
- **`latex/`**: LaTeX guides and manuscripts detailing the sensitivity extensions.

---

## 🚀 Getting Started

### Installation
Ensure you have **Python 3.12** installed.

```bash
cd core
pip install -r requirements.txt
```

### Running the Core Package
To reproduce every result, table, and figure from the core paper:
```bash
cd core
bash scripts/run_all.sh
```

### Running Sensitivity Analysis
To run the policy experiments, navigate to the sensitivity code folder and run the `policy_experiment.py` script with the desired flag:

```bash
cd sensitivity/code
# Examples:
python policy_experiment.py base        # Year-level couplings
python policy_experiment.py maps       # Anomaly-score sensitivity
python policy_experiment.py regional   # Regional G-STA and lead-lag
python policy_experiment.py mitigation   # Counterfactual (i)
```

---

## 📜 Citation
If you use this code in your research, please cite the original paper: 
*"Graph Spatio-Temporal Alignment for Trajectories of Directed, Partially Observed Attributed Networks"*
