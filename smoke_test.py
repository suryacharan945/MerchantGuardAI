"""
MerchantGuard AI - System Integrity Smoke Test
Verifies dataset loading, feature alignment, LightGBM inference, Platt calibration,
FAISS vector retrieval, and persistent SQLite audit writes.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def run_smoke_test():
    print("==================================================")
    print("🛡️  RUNNING MERCHANTGUARD AI SMOKE TEST")
    print("==================================================")

    # 1. Verify Dataset Splits
    split_path = os.path.join(PROJECT_ROOT, "data/splits/test_temporal_full.csv")
    assert os.path.exists(split_path), f"Missing holdout split: {split_path}"
    df_test = pd.read_csv(split_path)
    print(f"✅ 1. Test Dataset Verified: {len(df_test)} transactions loaded.")

    # 2. Verify Models & Feature Columns
    model_path = os.path.join(PROJECT_ROOT, "models/lightgbm_temporal.pkl")
    cal_path = os.path.join(PROJECT_ROOT, "models/calibrator_platt_temporal.pkl")
    cols_path = os.path.join(PROJECT_ROOT, "data/splits/feature_columns_v3b.json")
    
    assert os.path.exists(model_path), "Missing LightGBM model artifact."
    assert os.path.exists(cal_path), "Missing Platt calibrator artifact."
    assert os.path.exists(cols_path), "Missing feature schema."

    model = joblib.load(model_path)
    calibrator = joblib.load(cal_path)
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)
    print(f"✅ 2. Model Pipeline Verified: LightGBM + Platt Calibrator loaded ({len(feature_cols)} features).")

    # 3. Model Inference & Calibration Check
    sample_rows = df_test[feature_cols].head(5)
    raw_scores = model.predict(sample_rows)
    calibrated_probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    
    assert len(calibrated_probs) == 5, "Inference output size mismatch."
    assert (calibrated_probs >= 0.0).all() and (calibrated_probs <= 1.0).all(), "Calibrated probabilities out of [0, 1] bounds."
    print(f"✅ 3. Inference Verified: 5 transactions scored. Prob range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}].")

    # 4. Policy RAG Retrieval Check
    from src.rag.retriever import PolicyRetriever
    retriever = PolicyRetriever(PROJECT_ROOT)
    test_tx = df_test.iloc[0].to_dict()
    matches = retriever.retrieve_context(test_tx, [("amount_vs_avg_ratio", 0.5)], top_k=1)
    assert len(matches) > 0, "Policy retrieval returned empty results."
    print(f"✅ 4. Policy RAG Verified: Retained [{matches[0]['doc_id']}] '{matches[0]['title']}'.")

    # 5. Persistent Audit SQLite Check
    from src.agent.audit_db import AuditDB
    db_path = os.path.join(PROJECT_ROOT, "reports/smoke_test_audit.db")
    audit = AuditDB(db_path)
    audit.log_event({
        "tx_id": "smoke_test_001",
        "amount": 4200.0,
        "calibrated_risk": float(calibrated_probs[0]),
        "prescribed_action": "HOLD",
        "confidence": "high",
        "sop_id": matches[0]["doc_id"],
        "rationale": "Verified during automated smoke test run.",
        "executed_tool": "settlement.place_escrow_hold()"
    })
    recent = audit.fetch_recent(limit=1)
    assert len(recent) == 1 and recent.iloc[0]["tx_id"] == "smoke_test_001", "SQLite write failed."
    print(f"✅ 5. Audit Persistence Verified: SQLite transaction written and fetched.")

    print("==================================================")
    print("🎉 ALL SMOKE TESTS PASSED: SYSTEM IS SUBMISSION-READY")
    print("==================================================")

if __name__ == "__main__":
    run_smoke_test()