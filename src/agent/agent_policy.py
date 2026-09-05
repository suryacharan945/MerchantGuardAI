import json

class AutonomousAgentPolicy:
    """
    Deterministic Agent Triage policy enforcing safe-mode limits.
    Prevents unauthorized or out-of-boundary remediation execution.
    """
    def __init__(self, high_confidence_threshold=0.85, soft_threshold=0.59):
        self.high_conf_threshold = high_confidence_threshold
        self.soft_threshold = soft_threshold

    def evaluate_triage(self, evidence_pack: dict) -> dict:
        telemetry = evidence_pack["telemetry"]
        score = telemetry["calibrated_risk_score"]
        ratio = telemetry["amount_vs_avg_ratio"]
        velocity = telemetry["device_tx_count_1h"]
        sops = evidence_pack.get("compliance_sops", [])
        
        sop_rule = sops[0].get("action_rule", "HOLD") if sops else "HOLD"
        sop_id = sops[0].get("sop_id", sops[0].get("doc_id", "SOP-GEN")) if sops else "SOP-GEN"

        # Determine confidence level
        if score >= self.high_conf_threshold or score <= 0.05:
            confidence = "high"
        elif score >= self.soft_threshold:
            confidence = "medium"
        else:
            confidence = "low"

        # Policy decision tree
        if velocity >= 5 or sop_rule == "BLOCK":
            recommended_action = "BLOCK"
            rationale = f"Severe burst card-testing detected ({velocity} tx/hr from device). Immediate block enforced under {sop_id}."
        elif ratio >= 3.0 or score >= self.soft_threshold:
            recommended_action = "HOLD" if score >= 0.75 else "OTP_VERIFY"
            rationale = f"High spending spike ({ratio:.1f}x baseline). Gated under {sop_id} to protect merchant dispute ledger."
        elif score >= 0.15 or telemetry["ip_country_mismatch"] == 1:
            recommended_action = "OTP_VERIFY"
            rationale = f"Medium risk cross-border or off-hours mismatch. Initiating challenge flow per {sop_id}."
        else:
            recommended_action = "ALLOW"
            rationale = "Transaction conforms to verified 30-day customer behavior profile."

        # Safety Circuit Breaker: Non-destructive gating for high-impact amounts
        can_auto_execute = (confidence == "high")
        if recommended_action == "BLOCK" and telemetry["amount_inr"] > 10000:
            can_auto_execute = False
            rationale += " [SAFETY BREAKER: Large block amount requires human analyst confirmation]."

        return {
            "recommended_action": recommended_action,
            "confidence": confidence,
            "can_auto_execute": can_auto_execute,
            "rationale": rationale,
            "sop_id": sop_id
        }
