
import numpy as np
from scipy.stats import ks_2samp

def detect_drift(reference_df, current_df, feature_cols, alpha=0.05):
    """Per-feature KS two-sample test. NOTE: KS compares marginal value
    distributions — it will NOT detect shifts in circular/cyclical features
    (e.g. hour-of-day) if the shift preserves the marginal shape (a uniform
    distribution shifted circularly is still uniform). Circular features
    need a dedicated circular-statistics test (e.g. comparing shifted
    angular means) if that failure mode matters for your monitoring needs."""
    results = []
    for col in feature_cols:
        stat, p_value = ks_2samp(reference_df[col], current_df[col])
        results.append({
            "feature": col, "ks_statistic": float(stat),
            "p_value": float(p_value), "drift_detected": bool(p_value < alpha)
        })
    return results

def drift_summary(drift_results, strong_evidence_p=0.001, min_features_for_soft_alert=2):
    """v2: alerts on EITHER condition —
    (a) any single feature with very strong evidence (p < strong_evidence_p,
        e.g. amount shifting 35% with p=0.00003), OR
    (b) multiple features crossing the standard alpha threshold together
        (corroborating, weaker-individually evidence).
    v1 only checked (b), which silently missed a severe single-feature
    shift like the amount-inflation case."""
    n_drifted = sum(r["drift_detected"] for r in drift_results)
    drifted_features = [r["feature"] for r in drift_results if r["drift_detected"]]
    strong_single_feature = [r["feature"] for r in drift_results if r["p_value"] < strong_evidence_p]

    alert = (len(strong_single_feature) >= 1) or (n_drifted >= min_features_for_soft_alert)
    alert_reason = (
        f"Strong single-feature drift: {strong_single_feature}" if strong_single_feature
        else f"{n_drifted} features drifted together" if n_drifted >= min_features_for_soft_alert
        else "No alert condition met"
    )

    return {
        "n_features_checked": len(drift_results),
        "n_features_drifted": n_drifted,
        "drift_rate": n_drifted / len(drift_results),
        "drifted_features": drifted_features,
        "strong_single_feature_drift": strong_single_feature,
        "alert": alert,
        "alert_reason": alert_reason
    }
