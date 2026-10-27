#!/bin/bash
# Runs the simulation studies sequentially (single core); logs in results/
cd "$(dirname "$0")/.."
python3 scripts/run_verification.py   > results/log_verification.txt 2>&1
python3 scripts/run_sim_detection.py  > results/log_sim_detection.txt 2>&1
python3 scripts/run_sim_missing.py    > results/log_sim_missing.txt 2>&1
python3 scripts/run_sim_leadlag.py    > results/log_sim_leadlag.txt 2>&1
echo DONE > results/SIMS_DONE
