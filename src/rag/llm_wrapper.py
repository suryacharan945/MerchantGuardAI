import os
import json
import hashlib
import re

class HardenedLLMWrapper:
    def __init__(self, cache_file: str, api_key: str = None):
        self.cache_file = cache_file
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.cache = self._load_cache()
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _persist_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def redact_payload(self, text_payload: str) -> str:
        """Enforces rigorous server-side PII sanitization."""
        # Mask email addresses
        text_payload = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text_payload)
        # Mask 16-digit card / PAN numbers
        text_payload = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_NUMBER_REDACTED]', text_payload)
        # Mask Indian mobile numbers
        text_payload = re.sub(r'\b(?:\+91|0)?[6-9]\d{9}\b', '[PHONE_REDACTED]', text_payload)
        return text_payload

    def generate(self, system_instruction: str, user_payload: dict, fallback_generator=None) -> dict:
        raw_user_str = json.dumps(user_payload, sort_keys=True)
        sanitized_user_str = self.redact_payload(raw_user_str)
        
        # Compute deterministic cache fingerprint
        prompt_hash = hashlib.md5(f"{system_instruction}:{sanitized_user_str}".encode()).hexdigest()
        
        if prompt_hash in self.cache:
            res = dict(self.cache[prompt_hash])
            res["generation_source"] = "deterministic_cache"
            return res

        # Try Live Gemini Call if available
        if self.client:
            try:
                from google.genai import types
                combined_prompt = f"{system_instruction}\n\nInput Payload:\n{sanitized_user_str}"
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=combined_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=300
                    )
                )
                output = json.loads(response.text.strip())
                output["generation_source"] = "live_gemini"
                
                self.cache[prompt_hash] = output
                self._persist_cache()
                return output
            except Exception:
                pass

        # Offline Fallback Generator
        if fallback_generator:
            output = fallback_generator(user_payload)
        else:
            output = {
                "explanation": "High velocity or spending spike detected against merchant baseline.",
                "recommended_action": "HOLD",
                "action_reason": "Automated security rule triggered under active operational SOP.",
                "policy_doc_id": "SOP-GEN-001"
            }
            
        output["generation_source"] = "offline_template_fallback"
        self.cache[prompt_hash] = output
        self._persist_cache()
        return output