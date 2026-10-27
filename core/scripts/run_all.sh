#!/bin/bash
# Reproduces every result, table and figure of the paper and supplement (about 1.5 h on one core).
# All stages are deterministic and resumable: interrupted runs continue from results/parts/.
set -e
cd "$(dirname "$0")/.."
python3 tests/test_identities.py
python3 scripts/run_verification.py
python3 scripts/run_verification2.py
python3 scripts/run_benchmark.py
VARIANT=base,N24,sigma0.4,kappa1.0,masssd0.5,relabel python3 scripts/run_sim_detection.py
python3 scripts/run_sim_leadlag2.py
python3 scripts/run_sim_missing.py
python3 scripts/run_empirical.py e0 e1 e2 e3 e6
E5_GEOMS=directed,symmetrised python3 scripts/run_empirical.py e5
E4_B=30 python3 scripts/run_empirical.py e4
python3 scripts/make_tables.py
python3 figures/make_figures.py
