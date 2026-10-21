# Balanced LightGBM with a CV Threshold vs. Random Forest on PC4 (Stand-in Experiment)

Motivation: In "WebRTC testing: challenges and practical solutions" (2017), testing a large real-time communication codebase is costly, so there is strong interest in predicting which units of a test run are likely to reveal failures and prioritising them. This repository tests a simple version of that idea on public data.

Question: Can a class-balanced LightGBM model with a cross-validated decision threshold flag the failure-prone units of a test run better than a standard random forest?

Dataset: OpenML PC4 (data id 1050, fetched via scikit-learn), a software-defect dataset with static code metrics and a defect label. It is a proxy for failure-prone units, not WebRTC test-run logs, and it is not the paper's own dataset.

Baseline vs. improved:
- Baseline: standard scikit-learn RandomForestClassifier (300 trees, default settings, threshold 0.5).
- Improved: LightGBM with balanced class weights, hyperparameters chosen by 5-fold cross-validation on the training split, and a decision threshold chosen by cross-validated F1 on the training split.
- Main metric: F1 for the defective class on the held-out test split. Secondary: ROC-AUC and Brier score.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (and `figures/f1_comparison.png`).

## Results (latest run)

On the public OpenML PC4 defect proxy dataset, a standard random forest classifier reached 40% F1 versus 61% for a class-balanced LightGBM with a CV-tuned threshold, though ROC-AUC barely moved (0.932 vs 0.936) on only 45 test positives.

| Metric | Value |
|---|---|
| baseline_score | 0.4039 |
| improved_score | 0.6139 |
| improvement | 0.2099 |
| baseline_score_std | 0.01018 |
| improved_score_std | 0 |
| baseline_f1 | 0.4039 |
| improved_f1 | 0.6139 |
| baseline_roc_auc | 0.9323 |
| improved_roc_auc | 0.9358 |
| improvement_roc_auc | 0.003484 |
| baseline_brier | 0.0685 |
| improved_brier | 0.08908 |
| improvement_brier | -0.02059 |
| decision_threshold | 0.71 |
| test_positives | 45 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Proxy dataset: PC4 static code metrics predict defect-prone modules, not WebRTC test-run failures. Only 45 positives in the test split, so F1 is high-variance; a single split is used. Model seeds vary only model randomness, not the split. Selected LightGBM params: {'learning_rate': 0.03, 'min_child_samples': 5, 'n_estimators': 100, 'num_leaves': 7}; threshold 0.71. The split uses seed 42 (house rule) rather than the seed 0 in the plan. fetch_openml has no timeout argument, so retries wrap the call instead.

## Reference

This project is motivated by the following paper:

> B Garcia, F Gortazar (2017). *WebRTC testing: challenges and practical solutions*. IEEE. https://ieeexplore.ieee.org/abstract/document/7992926/

Paper author: Prof. Dr. Francisco Gortázar Bellas, Universidad Rey Juan Carlos (francisco.gortazar@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
