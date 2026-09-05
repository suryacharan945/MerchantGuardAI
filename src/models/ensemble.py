
import numpy as np
from sklearn.linear_model import LogisticRegression

def normalize_scores(scores, fixed_min=None, fixed_max=None):
    """Min-max normalize to [0,1]. CRITICAL: pass fixed_min/fixed_max (from
    normalization_bounds.json, computed once on the training distribution)
    when scoring ANY batch smaller than the full reference set — especially
    single transactions. Without fixed bounds, min==max for a 1-row input
    and this silently returns 0.0 regardless of the actual score. Only omit
    fixed_min/fixed_max when normalizing a full, representative batch where
    batch-relative scaling is intentional (e.g. bulk evaluation reports)."""
    if fixed_min is not None and fixed_max is not None:
        lo, hi = fixed_min, fixed_max
    else:
        lo, hi = scores.min(), scores.max()
    normalized = (scores - lo) / max(hi - lo, 1e-8)
    return np.clip(normalized, 0.0, 1.0)

def combine_novelty_signals(ae_scores, if_scores, bounds=None):
    if bounds:
        ae_norm = normalize_scores(ae_scores, bounds["autoencoder"]["min"], bounds["autoencoder"]["max"])
        if_norm = normalize_scores(if_scores, bounds["isolation_forest"]["min"], bounds["isolation_forest"]["max"])
    else:
        ae_norm = normalize_scores(ae_scores)
        if_norm = normalize_scores(if_scores)
    return (ae_norm + if_norm) / 2

def ensemble_score(lgb_scores, ae_scores, if_scores, lgb_weight=0.65, bounds=None):
    """bounds: dict loaded from models/normalization_bounds.json. REQUIRED
    for single-transaction scoring (Streamlit, manual checks) — without it,
    any single transaction will silently score near-zero regardless of
    actual risk. Optional for full-batch evaluation reports."""
    if bounds:
        lgb_norm = normalize_scores(lgb_scores, bounds["lightgbm"]["min"], bounds["lightgbm"]["max"])
    else:
        lgb_norm = normalize_scores(lgb_scores)
    novelty = combine_novelty_signals(ae_scores, if_scores, bounds)
    return lgb_weight * lgb_norm + (1 - lgb_weight) * novelty

def calibrate_platt(raw_ensemble_scores, y_true):
    calibrator = LogisticRegression()
    calibrator.fit(raw_ensemble_scores.reshape(-1, 1), y_true)
    return calibrator

def apply_platt(calibrator, raw_scores):
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
