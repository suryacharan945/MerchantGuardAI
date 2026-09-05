import os
import json
import re

class DualModeLLMGenerator:
    """
    Dual-mode explanation generator:
    - Primary: Google Gemini (gemini-2.5-flash) via google-genai SDK with temperature=0.0.
    - Fallback: Deterministic template engine if API key is missing or offline.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.mode = "api_gemini"
            except Exception as e:
                print(f"⚠️ Failed to initialize GenAI client: {e}. Defaulting to offline fallback.")
                self.mode = "offline_fallback"
        else:
            self.mode = "offline_fallback"

    def _fallback_generate(self, prompt_bundle: dict) -> dict:
        """Deterministic, rule-based fallback generating compliant JSON without API calls."""
        payload = json.loads(prompt_bundle["user"])
        telemetry = payload.get("transaction_telemetry", {})
        shap_list = payload.get("top_shap_attributions", [])
        policies = payload.get("applicable_sops", [])
        
        amt = telemetry.get("amount_inr", 0)
        ratio = telemetry.get("amount_vs_avg_ratio", 1.0)
        prob = telemetry.get("calibrated_fraud_probability", 0.0)
        
        top_sop = policies[0] if policies else {
            "doc_id": "SOP-GEN-000",
            "mandated_action": "HOLD",
            "title": "Standard Risk Gate"
        }
        
        # Build explanation from top SHAP features
        reasons = []
        for item in shap_list[:2]:
            feat = item["feature"]
            if feat == "amount" or feat == "amount_vs_avg_ratio":
                reasons.append(f"spend is {ratio:.1f}x typical customer baseline (INR {amt:,.0f})")
            elif "device" in feat:
                reasons.append("transaction originates from a novel or burst device fingerprint")
            elif "off_hours" in feat:
                reasons.append("payment occurred during elevated off-hours")
            elif "ip" in feat:
                reasons.append("cross-border IP mismatch detected")
            else:
                reasons.append(f"elevated risk signal from {feat.replace('_', ' ')}")
                
        if not reasons:
            reasons = ["statistical deviation across behavioral parameters"]
            
        explanation = f"Flagged as risk probability reached {prob*100:.1f}%: {'; '.join(reasons)}."
        
        # Map action
        mandated = top_sop.get("mandated_action", "HOLD")
        if mandated == "HOLD_OR_OTP":
            action = "HOLD" if (ratio >= 3.0 or prob > 0.8) else "OTP_VERIFY"
        elif mandated in ["BLOCK", "HOLD", "OTP_VERIFY", "ALLOW"]:
            action = mandated
        else:
            action = "HOLD"

        return {
            "explanation": explanation[:160],
            "recommended_action": action,
            "action_reason": f"Aligned with {top_sop['doc_id']} ({top_sop['title']}) given calibrated risk.",
            "policy_doc_id": top_sop["doc_id"],
            "generation_mode": "offline_fallback"
        }

    def generate(self, prompt_bundle: dict) -> dict:
        """Executes API generation or routes to offline fallback."""
        if self.mode == "offline_fallback" or self.client is None:
            return self._fallback_generate(prompt_bundle)
            
        try:
            from google.genai import types
            
            prompt_content = f"{prompt_bundle['system']}\n\nInput Payload:\n{prompt_bundle['user']}"
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=300
                )
            )
            
            raw_text = response.text.strip()
            parsed = json.loads(raw_text)
            parsed["generation_mode"] = "api_gemini"
            return parsed
            
        except Exception as e:
            # Resilient fallback on network timeout, quota exhaustion, or parsing error
            fallback_res = self._fallback_generate(prompt_bundle)
            fallback_res["generation_mode"] = f"offline_fallback_on_error: {str(e)[:40]}"
            return fallback_res
