"""Stand-in experiment: class-balanced LightGBM with a CV-chosen threshold vs. a standard
random forest, on OpenML PC4 (software-defect data used as a proxy for failure-prone units)."""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

SEED = 42                      # split and CV seed
SEEDS = [42, 43, 44]           # model seeds for the randomized models
MAX_ROWS = 200_000             # subsample cap to keep runtime small
THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def fetch_pc4(retries=4):
    """Download OpenML PC4 with retry and exponential backoff on transient errors."""
    for attempt in range(retries):
        try:
            return fetch_openml(name="pc4", as_frame=True, parser="auto")
        except Exception as exc:  # network / HTTP 429 / 5xx surface as generic exceptions here
            wait = 2 ** attempt
            print(f"Download failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Could not download PC4 from OpenML")


def load_data():
    """Return numeric feature matrix and binary label (1 = defect-prone, the minority class)."""
    bunch = fetch_pc4()
    X = bunch.data.apply(pd.to_numeric, errors="coerce").astype(float)
    labels = bunch.target.astype(str)
    minority = labels.value_counts().index[-1]  # defective modules are the rare class in PC4
    y = (labels == minority).astype(int).to_numpy()
    if len(X) > MAX_ROWS:  # subsample very large data (not triggered for PC4)
        idx = X.sample(MAX_ROWS, random_state=SEED).index
        X, y = X.loc[idx], y[X.index.get_indexer(idx)]
    return X, y


def best_threshold(y, prob):
    """Threshold maximizing F1 on the given (out-of-fold) predictions."""
    scores = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def evaluate(y_true, prob, thr):
    """Shared evaluation: F1 (main), ROC-AUC and Brier (secondary) on the test split."""
    pred = (prob >= thr).astype(int)
    return {"f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, prob),
            "brier": brier_score_loss(y_true, prob)}


def main():
    X, y = load_data()
    print(f"Loaded PC4: {X.shape[0]} rows, {X.shape[1]} features, positive rate {y.mean():.3f}")

    # Single stratified 75/25 split; test set is touched only at the end.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    med = Xtr.median()  # shared preprocessing: median imputation fitted on training data
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    print(f"Split: {len(ytr)} train / {len(yte)} test")

    # Improved model: hyperparameters chosen by 5-fold CV ROC-AUC on the training split only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = {"num_leaves": [7, 15], "learning_rate": [0.03, 0.1],
            "n_estimators": [100, 300], "min_child_samples": [5, 20]}
    search = GridSearchCV(LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1,
                                         n_jobs=-1),
                          grid, scoring="roc_auc", cv=cv, n_jobs=1)
    search.fit(Xtr, ytr)
    best = search.best_estimator_
    print(f"Best LightGBM params (CV ROC-AUC {search.best_score_:.3f}): {search.best_params_}")

    # Decision threshold chosen by CV F1 using out-of-fold probabilities on the training split.
    oof = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr = best_threshold(ytr, oof)
    print(f"Chosen decision threshold (CV F1): {thr:.2f}")

    # Test evaluation, once per seed for both models.
    base_runs, imp_runs = [], []
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1).fit(Xtr, ytr)
        base_runs.append(evaluate(yte, rf.predict_proba(Xte)[:, 1], 0.5))
        lgb = clone(best).set_params(random_state=s).fit(Xtr, ytr)
        imp_runs.append(evaluate(yte, lgb.predict_proba(Xte)[:, 1], thr))
        print(f"seed {s}: baseline F1={base_runs[-1]['f1']:.3f}  improved F1={imp_runs[-1]['f1']:.3f}")

    def agg(runs, key):
        vals = np.array([r[key] for r in runs], dtype=float)
        return float(vals.mean()), float(vals.std())

    b_f1, b_f1_sd = agg(base_runs, "f1")
    i_f1, i_f1_sd = agg(imp_runs, "f1")
    b_auc, _ = agg(base_runs, "roc_auc")
    i_auc, _ = agg(imp_runs, "roc_auc")
    b_br, _ = agg(base_runs, "brier")
    i_br, _ = agg(imp_runs, "brier")

    # Sanity check: refuse to report meaningless metrics.
    vals = [b_f1, i_f1, b_auc, i_auc, b_br, i_br]
    if yte.sum() == 0 or not np.all(np.isfinite(vals)):
        print("ERROR: metrics not meaningful")
        sys.exit(1)

    metrics = {
        "baseline_score": b_f1, "improved_score": i_f1,
        "improvement": i_f1 - b_f1,
        "baseline_score_std": b_f1_sd, "improved_score_std": i_f1_sd,
        "baseline_f1": b_f1, "improved_f1": i_f1,
        "baseline_roc_auc": b_auc, "improved_roc_auc": i_auc,
        "improvement_roc_auc": i_auc - b_auc,
        "baseline_brier": b_br, "improved_brier": i_br,
        "improvement_brier": b_br - i_br,  # positive = improved has lower Brier
        "decision_threshold": thr,
        "test_positives": int(yte.sum()),
    }

    results = {
        "question": "Can a class-balanced LightGBM model with a cross-validated decision threshold "
                    "flag failure-prone units better than a standard random forest?",
        "method": "Stand-in on OpenML PC4 defect data. Single stratified 75/25 split (seed 42). "
                  "Baseline: RandomForest (300 trees, threshold 0.5). Improved: balanced LightGBM "
                  "with 5-fold CV hyperparameter search (ROC-AUC) and CV-F1 threshold, both on "
                  "training data only. Model seeds 42-44; mean and std reported.",
        "dataset": {"name": "OpenML PC4 (pc4, data id 1050)",
                    "source_url": "https://www.openml.org/d/1050",
                    "n_rows": int(len(X)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "F1 for defective class on test split",
        "higher_is_better": True,
        "baseline": "Standard scikit-learn RandomForestClassifier (300 trees, default settings, "
                    "0.5 decision threshold)",
        "improved": "LightGBM with class_weight='balanced'; hyperparameters chosen by 5-fold CV "
                    "ROC-AUC on the training split; decision threshold chosen by out-of-fold F1 "
                    "on the training split",
        "metrics": metrics,
        "notes": f"Proxy dataset: PC4 static code metrics predict defect-prone modules, not "
                 f"WebRTC test-run failures. Only {int(yte.sum())} positives in the test split, so F1 "
                 f"is high-variance; a single split is used. Model seeds vary only model randomness, "
                 f"not the split. Selected LightGBM params: {search.best_params_}; threshold "
                 f"{thr:.2f}. The split uses seed 42 (house rule) rather than the seed 0 in the "
                 f"plan. fetch_openml has no timeout argument, so retries wrap the call instead.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Optional chart: baseline vs improved F1 with seed-to-seed spread.
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline RF", "Improved LGBM"], [b_f1, i_f1], yerr=[b_f1_sd, i_f1_sd], capsize=6,
           color=["grey", "steelblue"])
    ax.set_ylabel("Test F1 (defective class)")
    ax.set_title("PC4 proxy: mean F1 over seeds 42-44")
    fig.tight_layout()
    fig.savefig("figures/f1_comparison.png", dpi=120)
    print("Wrote results.json and figures/f1_comparison.png")


if __name__ == "__main__":
    main()
