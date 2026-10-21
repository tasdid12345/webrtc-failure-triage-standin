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

## Reference

This project is motivated by the following paper:

> B Garcia, F Gortazar (2017). *WebRTC testing: challenges and practical solutions*. IEEE. https://ieeexplore.ieee.org/abstract/document/7992926/

Paper author: Prof. Dr. Francisco Gortázar Bellas, Universidad Rey Juan Carlos (francisco.gortazar@urjc.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
