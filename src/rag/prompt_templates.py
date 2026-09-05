import json

def build_merchant_rag_prompt(sanitized_tx: dict, shap_top: list, retrieved_policies: list, calibrated_score: float) -> dict:
    """Builds a deterministic, structured JSON prompt for LLM consumption."""
    
    system_instruction = (
        "You are an AI Fraud Intelligence Assistant for Razorpay merchants. "
        "Your role is to analyze suspicious transactions using feature attributions (SHAP) "
        "and internal Standard Operating Procedures (SOPs). "
        "Return ONLY a strictly valid JSON object without markdown formatting or commentary."
    )
    
    shap_summary = [
        {"feature": feat, "importance": round(val, 4)} 
        for feat, val in shap_top if val > 0
    ]
    
    policy_excerpts = [
        {
            "doc_id": p["doc_id"],
            "title": p["title"],
            "guideline": p["text"],
            "mandated_action": p["action_rule"]
        } for p in retrieved_policies
    ]
    
    user_payload = {
        "transaction_telemetry": {
            "tx_id": sanitized_tx.get("tx_id", "N/A"),
            "merchant_id": sanitized_tx.get("merchant_id", "N/A"),
            "amount_inr": sanitized_tx.get("amount", 0),
            "amount_vs_avg_ratio": round(sanitized_tx.get("amount_vs_avg_ratio", 1.0), 2),
            "payment_method": sanitized_tx.get("payment_method", "N/A"),
            "user_id_hash": sanitized_tx.get("user_id", "N/A"),
            "device_hash": sanitized_tx.get("device_fingerprint", "N/A"),
            "calibrated_fraud_probability": round(calibrated_score, 4)
        },
        "top_shap_attributions": shap_summary,
        "applicable_sops": policy_excerpts,
        "required_output_schema": {
            "explanation": "A concise 1-2 sentence merchant-facing explanation (max 35 words).",
            "recommended_action": "One of [ALLOW, OTP_VERIFY, HOLD, BLOCK]",
            "action_reason": "One concise sentence referencing the matching SOP.",
            "policy_doc_id": "ID of the most relevant SOP applied."
        }
    }
    
    return {
        "system": system_instruction,
        "user": json.dumps(user_payload, indent=2)
    }
