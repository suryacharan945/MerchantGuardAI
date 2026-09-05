
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, precision_recall_curve
)

def train_lightgbm(X_train, y_train, X_test, y_test, params_override=None):
    """Train LightGBM with class-imbalance handling via scale_pos_weight."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 10,   # lower default — small dataset needs this
        "seed": 42
    }
    if params_override:
        params.update(params_override)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    return model, params

def precision_at_k(y_true, y_scores, k):
    """Precision among the top-k highest-scored transactions."""
    order = np.argsort(-y_scores)[:k]
    return y_true.iloc[order].sum() / k if hasattr(y_true, "iloc") else y_true[order].sum() / k

def recall_at_fpr(y_true, y_scores, target_fpr=0.01):
    """Recall achieved at (approximately) the given false-positive rate."""
    thresholds = np.linspace(0, 1, 1000)
    best_recall, best_fpr = 0, None
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tn = ((preds == 0) & (y_true == 0)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        tp = ((preds == 1) & (y_true == 1)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        if fpr <= target_fpr and recall > best_recall:
            best_recall, best_fpr = recall, fpr
    return best_recall, best_fpr

def evaluate(model, X_test, y_test):
    y_scores = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, y_scores)

    n_fraud_test = int(y_test.sum())
    k_values = [10, 20] if n_fraud_test < 30 else [50, 100]

    p_at_k = {f"precision_at_{k}": precision_at_k(y_test, y_scores, k) for k in k_values}

    recall_1pct, achieved_fpr = recall_at_fpr(y_test, y_scores, target_fpr=0.01)

    preds_binary = (y_scores >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds_binary, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds_binary)
    fp_per_1000 = (cm[0][1] / len(y_test)) * 1000

    return {
        "auroc": float(auc),
        **p_at_k,
        "recall_at_1pct_fpr": float(recall_1pct),
        "achieved_fpr": float(achieved_fpr) if achieved_fpr is not None else None,
        "precision_default_threshold": float(prec),
        "recall_default_threshold": float(rec),
        "f1_default_threshold": float(f1),
        "false_positives_per_1000": float(fp_per_1000),
        "confusion_matrix": cm.tolist(),
        "n_test_fraud": n_fraud_test,
        "k_values_used": k_values
    }, y_scores


def find_optimal_threshold(y_true, y_scores):
    """Pick the threshold maximizing F1 — appropriate for imbalanced fraud data
    where a fixed 0.5 threshold is usually miscalibrated."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

def evaluate_with_optimal_threshold(y_true, y_scores):
    threshold, best_f1 = find_optimal_threshold(y_true, y_scores)
    preds = (y_scores >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "optimal_threshold": threshold,
        "precision_at_optimal": float(precision),
        "recall_at_optimal": float(recall),
        "f1_at_optimal": float(best_f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn)
    }
