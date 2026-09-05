import json
import numpy as np

def build_evidence_package(tx_dict: dict, calibrated_score: float, shap_top: list, similar_cases: list, retrieved_sops: list) -> dict:
    """
    Consolidates model telemetry, explainability signals, policy SOPs,
    and historical twin cases into a standardized machine-readable evidence package.
    """
    evidence = {
        "telemetry": {
            "tx_id": tx_dict.get("tx_id", "anon_tx"),
            "merchant_id": tx_dict.get("merchant_id", "m_unknown"),
            "user_id_masked": f"usr_{str(tx_dict.get('user_id', 'anon'))[-6:]}",
            "amount_inr": float(tx_dict.get("amount", 0.0)),
            "amount_vs_avg_ratio": round(float(tx_dict.get("amount_vs_avg_ratio", 1.0)), 2),
            "device_tx_count_1h": int(tx_dict.get("device_tx_count_1h", 0)),
            "is_off_hours": int(tx_dict.get("is_off_hours", 0)),
            "ip_country_mismatch": int(tx_dict.get("ip_country_mismatch", 0)),
            "calibrated_risk_score": round(float(calibrated_score), 4)
        },
        "explainability_shap": [
            {"factor": feat, "attribution_weight": round(val, 4)}
            for feat, val in shap_top if val > 0
        ],
        "historical_twin_cases": [
            {
                "case_id": case.get("tx_id", f"case_{i}"),
                "similarity_pct": case.get("similarity_pct", 85.0),
                "pattern": "card_testing" if case.get("amount", 100) < 50 else "high_amount"
            } for i, case in enumerate(similar_cases[:2])
        ],
        "compliance_sops": [
            {
                "sop_id": p.get("doc_id", "SOP-001"),
                "title": p.get("title", "Standard Risk Protocol"),
                "action_rule": p.get("action_rule", "HOLD")
            } for p in retrieved_sops
        ]
    }
    return evidence
