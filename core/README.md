# G-STA reproducibility package (version 1.1.0)

Code and data for "Graph Spatio-Temporal Alignment for Trajectories of Directed, Partially Observed Attributed Networks".

## Install
Python 3.12 with the versions in `requirements.txt` (`pip install -r requirements.txt`). POT is needed only for cross-checks.

## One command
`bash scripts/run_all.sh` runs the automated tests (`tests/test_identities.py`) and reproduces every result, table and figure (about 1.5 h on one core). All stages use fixed seeds and are resumable (intermediate blocks are written to `results/parts/`).

## Layout
- `gsta/` the method: `ufgw.py` (Algorithm 1: directed, masked, unbalanced entropic UFGW; generalized conditional gradient with safeguarded step; audit logging; stationarity residuals; deterministic multi-start portfolio with optional pair-seeded random starts; envelope gradients), `softdtw.py` (soft-DTW value/gradient, Delannoy numbers, lead-lag index, boundary correction), `core.py` (trajectory-level quantities), `data.py` (UNCTADstat panel construction, name harmonization, masks, alternative preprocessing incl. common support), `sim.py` (simulation designs and baselines).
- `scripts/` `run_verification.py`, `run_verification2.py` (explicit-tensor gradient check, exhaustive path enumeration, convergence audit, multi-start diagnostics), `run_benchmark.py` (cost breakdown, comparison with POT), `run_sim_detection.py`, `run_sim_leadlag2.py`, `run_sim_missing.py`, `run_empirical.py` (stages e0-e6), `make_tables.py` (every LaTeX table, from `results/`).
- `figures/make_figures.py` draws every figure from `results/` only (one function per figure, written to be portable to MATLAB).
- `results/` all numerical results used in the paper. `data/` the exact UNCTADstat extracts used. `data_download/` retrieval scripts; set `UNCTAD_CLIENT_ID` and `UNCTAD_CLIENT_SECRET` as environment variables (no credentials are stored in the code).
