import json

class IncidentReportGenerator:
    """
    Synthesizes transaction telemetry, SHAP attributions, and retrieved policy SOPs
    into an executive incident summary and multi-channel customer communications.
    """
    def __init__(self):
        pass

    def generate_investigation_summary(self, tx_dict: dict, shap_bullets: list, policy_doc: dict, action_decision: dict) -> str:
        """Generates a structured, audit-ready incident investigation markdown report."""
        tx_id = tx_dict.get("tx_id", "N/A")
        merchant_id = tx_dict.get("merchant_id", "N/A")
        amount = tx_dict.get("amount", 0)
        ratio = tx_dict.get("amount_vs_avg_ratio", 1.0)
        score = action_decision.get("calibrated_score", tx_dict.get("calibrated_score", 0.0))
        action = action_decision.get("recommended_action", "HOLD")
        sop_id = policy_doc.get("doc_id", "SOP-GEN-001")
        sop_title = policy_doc.get("title", "Standard Operating Procedure")
        
        # Format top attributions
        top_cues = []
        for feat, val in shap_bullets:
            if val > 0:
                top_cues.append(f"- **{feat.replace('_', ' ').title()}** (attribution weight: +{val:.3f})")
        attributions_str = "\n".join(top_cues) if top_cues else "- Standard variance threshold reached."

        summary_md = f"""# Incident Investigation Brief: {tx_id}

### 1. Executive Incident Overview
Transaction **{tx_id}** processed under merchant **{merchant_id}** was intercepted by MerchantGuard AI's hybrid ensemble with a calibrated risk score of **{score:.4f}** ({score*100:.1f}% empirical fraud probability). The assessed transaction value is **INR {amount:,.2f}**, representing an anomaly ratio of **{ratio:.2f}x** relative to the customer's historical 30-day baseline.

### 2. Forensic Signal Attribution
Automated SHAP feature attributions isolated the following primary indicators driving the anomaly score:
{attributions_str}

### 3. Regulatory & Policy Grounding
This pattern activates internal compliance protocol **[{sop_id}] {sop_title}**. The protocol establishes that anomalous velocity and spending spikes require defensive operational intervention to mitigate chargeback risk and prevent unauthorized payment rails compromise.

### 4. Operational Disposition & Recommended Next Steps
- **Assigned Policy Action:** `{action}`
- **Automated Routing:** {action_decision.get('action_reason', 'Aligned with operational SOP.')}
- **Analyst Directives:**
  1. Freeze outgoing order fulfillment while transaction remains in `{action}` state.
  2. Verify device possession status and match customer registration records against IP telemetry.
  3. Log incident verdict into the merchant security portal audit ledger.
"""
        return summary_md

    def generate_comms_templates(self, tx_dict: dict, action_decision: dict) -> dict:
        """Generates personalized, action-specific customer communication drafts."""
        tx_id = tx_dict.get("tx_id", "N/A")
        short_id = tx_id[:8] if tx_id != "N/A" else "TXN"
        amount = tx_dict.get("amount", 0)
        merchant = tx_dict.get("merchant_id", "Merchant")
        action = action_decision.get("recommended_action", "HOLD")

        if action == "BLOCK":
            sms = f"Razorpay Security Alert: Transaction of INR {amount:,.0f} at {merchant} (Ref: {short_id}) was declined due to unusual activity. If this was you, visit rzp.io/sec to verify your card."
            email = {
                "subject": f"Security Alert: Transaction Declined at {merchant} (Ref: {short_id})",
                "body": (
                    f"Dear Customer,\n\n"
                    f"A payment of INR {amount:,.2f} attempted at {merchant} was declined by our automated fraud detection system "
                    f"to protect your account against unauthorized transactions.\n\n"
                    f"If you did not initiate this transaction, please reach out to your card-issuing bank immediately to secure your instrument. "
                    f"If this was authorized by you, please complete one-time identity verification here: https://merchantguard.razorpay.com/verify/{short_id}.\n\n"
                    f"Regards,\nMerchant Risk & Fraud Security Team"
                )
            }
        elif action == "OTP_VERIFY":
            sms = f"Razorpay Security: Enter OTP 839-201 to authorize INR {amount:,.0f} for {merchant}. Valid for 5 minutes. Never share your OTP with anyone."
            email = {
                "subject": f"Action Required: Verify your transaction at {merchant}",
                "body": (
                    f"Dear Customer,\n\n"
                    f"A payment of INR {amount:,.2f} requires additional verification to complete checkout at {merchant}.\n\n"
                    f"Please submit the 6-digit one-time password (OTP) sent to your registered mobile device on the checkout screen. "
                    f"This additional step ensures your account remains protected against unauthorized card use.\n\n"
                    f"Regards,\nPayment Security Gateway"
                )
            }
        else:  # HOLD / REVIEW
            sms = f"Notice: Order {short_id} (INR {amount:,.0f}) at {merchant} is pending brief security review. Fulfillment will resume shortly. Support: rzp.io/help"
            email = {
                "subject": f"Notice: Payment Processing Update for Order #{short_id}",
                "body": (
                    f"Dear Customer,\n\n"
                    f"Your payment of INR {amount:,.2f} at {merchant} has been received and is undergoing standard security review.\n\n"
                    f"No further action is required from you at this time. Most reviews are resolved within 2-4 operational hours, "
                    f"after which your order fulfillment will proceed normally. We appreciate your patience as we ensure transaction safety.\n\n"
                    f"Regards,\nMerchant Operations Desk"
                )
            }

        return {
            "recommended_action": action,
            "sms_template": sms,
            "email_template": email
        }
