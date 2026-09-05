
from datetime import datetime
import uuid

ACTION_MAP = {
    "Low":      {"action": "Allow",         "auto_execute": True,  "requires_review": False},
    "Medium":   {"action": "OTP Verify",    "auto_execute": True,  "requires_review": False},
    "High":     {"action": "Hold",          "auto_execute": True,  "requires_review": True},
    "Critical": {"action": "Block",         "auto_execute": True,  "requires_review": True},
}

ACTION_MESSAGE_TEMPLATES = {
    "Allow": "Transaction approved automatically — risk profile consistent with normal activity.",
    "OTP Verify": "Please verify this transaction via OTP sent to your registered device before it completes.",
    "Hold": "Transaction placed on hold pending manual review. You will be notified within 24 hours.",
    "Block": "Transaction blocked due to high fraud risk. Contact merchant support if this was you.",
}

def recommend_action(severity, calibrated_score, explanation_text, tx_id=None):
    if severity not in ACTION_MAP:
        raise ValueError(f"Unknown severity bucket: {severity}")
    policy = ACTION_MAP[severity]
    action = policy["action"]
    return {
        "audit_id": str(uuid.uuid4()), "tx_id": tx_id,
        "timestamp": datetime.now().isoformat(),
        "calibrated_risk_score": float(calibrated_score),
        "severity": severity, "recommended_action": action,
        "requires_human_review": policy["requires_review"],
        "explanation": explanation_text,
        "merchant_message": ACTION_MESSAGE_TEMPLATES[action],
        "policy_version": "v1_severity_rule_table"
    }

def simulate_failure_handling(audit_entry, failure_type="notification_timeout"):
    fallback_map = {
        "notification_timeout": "OTP dispatch failed — falling back to Hold for manual review rather than silently allowing.",
        "model_unavailable": "Risk model unreachable — falling back to rule-based amount/velocity check (conservative default: Hold).",
    }
    audit_entry["failure_simulated"] = failure_type
    audit_entry["fallback_action"] = fallback_map.get(failure_type, "Hold (safe default)")
    audit_entry["original_action"] = audit_entry["recommended_action"]
    audit_entry["recommended_action"] = "Hold"
    return audit_entry


# --- Safety-net override (Step 8b) ---
# A single very large transaction can cause outsized loss even if the model's
# behavioral score alone doesn't flag it as Critical. This deterministic rule
# guarantees a minimum response for large amounts, independent of model
# confidence — an explicit "bounded and gated" business control, not a model
# decision. Threshold = 99th percentile of legit transaction amounts.
LARGE_AMOUNT_THRESHOLD = 3600.0
MIN_ACTION_FOR_LARGE_AMOUNT = "Hold"  # never below Hold, regardless of model score

ACTION_RANK = {"Allow": 0, "OTP Verify": 1, "Hold": 2, "Block": 3}

def apply_safety_net(audit_entry, amount):
    """Call this AFTER recommend_action() — enforces a floor, never lowers
    an action the model already recommended more strongly."""
    if amount >= LARGE_AMOUNT_THRESHOLD:
        current_rank = ACTION_RANK[audit_entry["recommended_action"]]
        min_rank = ACTION_RANK[MIN_ACTION_FOR_LARGE_AMOUNT]
        if current_rank < min_rank:
            audit_entry["safety_net_triggered"] = True
            audit_entry["pre_override_action"] = audit_entry["recommended_action"]
            audit_entry["recommended_action"] = MIN_ACTION_FOR_LARGE_AMOUNT
            audit_entry["override_reason"] = (
                f"Amount \u20b9{amount:,.0f} exceeds large-transaction threshold "
                f"(\u20b9{LARGE_AMOUNT_THRESHOLD:,.0f}) — minimum action enforced "
                f"regardless of model confidence."
            )
        else:
            audit_entry["safety_net_triggered"] = False
    else:
        audit_entry["safety_net_triggered"] = False
    return audit_entry


# --- Safety-net override v2 (Step 8c — replaces the v1 amount-only rule,
#     which was proven to add false-positive friction with zero fraud-catch
#     benefit, since missed fraud in testing had amounts BELOW the legit
#     99th-percentile threshold used in v1) ---
MODERATE_AMOUNT_THRESHOLD = 900.0
ACTION_RANK = {"Allow": 0, "OTP Verify": 1, "Hold": 2, "Block": 3}

def apply_safety_net_v2(audit_entry, amount, ip_country_mismatch=0, is_off_hours=0,
                          user_device_nunique=1):
    """v2: requires amount above a MODERATE (75th pct legit) threshold AND at
    least one behavioral red flag, rather than amount alone. This targets
    cases the model under-scored despite real corroborating evidence,
    instead of blanket-catching ordinary large legit purchases."""
    has_flag = bool(ip_country_mismatch) or bool(is_off_hours) or (user_device_nunique > 2)
    if amount >= MODERATE_AMOUNT_THRESHOLD and has_flag:
        current_rank = ACTION_RANK[audit_entry["recommended_action"]]
        min_rank = ACTION_RANK["OTP Verify"]  # softer floor than v1's Hold —
                                                # proportionate to weaker evidence
        if current_rank < min_rank:
            audit_entry["safety_net_triggered"] = True
            audit_entry["pre_override_action"] = audit_entry["recommended_action"]
            audit_entry["recommended_action"] = "OTP Verify"
            audit_entry["override_reason"] = (
                f"Amount \u20b9{amount:,.0f} above moderate threshold "
                f"(\u20b9{MODERATE_AMOUNT_THRESHOLD:,.0f}) with a corroborating "
                f"behavioral flag — minimum OTP Verify enforced."
            )
        else:
            audit_entry["safety_net_triggered"] = False
    else:
        audit_entry["safety_net_triggered"] = False
    return audit_entry
