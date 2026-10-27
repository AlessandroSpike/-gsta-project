# G-STA sensitivity analysis (new Sections 4.5, 5.5, 6.5; Supplement S8)

## latex/
- ALL_CHANGES_main.tex, ALL_CHANGES_supplement.tex: every chunk, with its location, the exact anchor
  text in your file and the mode (insert before / insert after / replace). These are insertion guides
  and do not compile on their own.
- main_with_sensitivity.tex, supplement_with_sensitivity.tex: your main__3_.tex and supplement__2_.tex
  with all chunks applied; both compile without LaTeX errors. The PDFs use placeholder images for your
  existing figures, which were not in the project; the new figures are real.
- Copy figures/*.pdf into the figures/ folder of your Overleaf project.

## code/
Run from a folder that contains the three CSV files:
    python policy_experiment.py base         # year-level couplings and Hessian certificates
    python policy_experiment.py maps         # anomaly-score sensitivity (Table 6, Tables S15-S16)
    python policy_experiment.py regional     # regional G-STA and lead-lag gradients (Table 7)
    python policy_experiment.py fd           # re-solved finite differences on the data (Table S14)
    python policy_experiment.py mitigation   # counterfactual (i)
    python policy_experiment.py stress       # counterfactual (ii)
    python policy_experiment.py timing       # counterfactual (iii)
    python policy_experiment.py uncertainty  # delta method vs Monte Carlo, masks (Table S18)
    python verify_sensitivity.py             # synthetic checks (Tables S12-S13, Figure S6)
    python make_outputs.py                   # figures and tables
gsta_core.py re-implements Algorithm 1 as described in the manuscript and reproduces Table S9, the
spectrum and anomaly ranks of Section 6 and Table 5. To use your own gsta solver instead, note that the
sensitivity layer needs only the selected couplings. Total runtime is about one hour on one core.

## results/
CSV files behind every number in the new tables.
