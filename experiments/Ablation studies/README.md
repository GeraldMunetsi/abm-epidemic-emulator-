# Ablation Studies

## Output layout (`out/`)

```
out/
├── fig_ablation_*.png          cross-condition comparison figures (step3_plot_training_curves.py)
├── plots/                      per-condition/validation plots (step2_validate.py, step3_plot_training_curves.py)
├── results/testing/<condition>/<strategy>/   test_final_statistics.json (src/step5_test.py)
└── trained-models/<condition>/ checkpoints, histories, replicate + validation reports
```
