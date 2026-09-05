import os
import json
import importlib.util

class PolicyRetriever:
    def __init__(self, project_root):
        self.project_root = project_root
        self.vs_script = os.path.join(project_root, "src/rag/vector_store.py")
        self.index_path = os.path.join(project_root, "models/policy_faiss.index")
        self.metadata_path = os.path.join(project_root, "models/policy_metadata.json")
        
        spec = importlib.util.spec_from_file_location("vector_store", self.vs_script)
        vs_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs_mod)
        
        self.store = vs_mod.PolicyVectorStore()
        self.store.load(self.index_path, self.metadata_path)

    def retrieve_context(self, tx_facts: dict, shap_bullets: list, top_k=2):
        """Constructs an intention-dense query from SHAP indicators and features."""
        risk_cues = []
        for feat, val in shap_bullets:
            if val > 0:
                risk_cues.append(feat.replace("_", " "))
        
        # Determine contextual severity anchors
        amount = tx_facts.get("amount", 0)
        ratio = tx_facts.get("amount_vs_avg_ratio", 1.0)
        off_hours = tx_facts.get("is_off_hours", 0)
        ip_mismatch = tx_facts.get("ip_country_mismatch", 0)
        
        contextual_intent = []
        if ratio >= 3.0 or amount > 5000:
            contextual_intent.append("High-value transaction spike security verification escalation")
        if off_hours:
            contextual_intent.append("off-hours transaction alert")
        if ip_mismatch:
            contextual_intent.append("cross-border IP mismatch anomaly")
        if "device_tx_count_1h" in [f[0] for f in shap_bullets]:
            contextual_intent.append("rapid multiple small velocity abuse card testing")
            
        search_str = f"{' '.join(contextual_intent)} | Key risk factors: {', '.join(risk_cues)}"
        
        raw_matches = self.store.search(search_str, top_k=top_k)
        
        # Safety gate: if risk signals are present, filter out SOP-LG-006 (Low-Risk Allow)
        if (ratio >= 2.0 or len(risk_cues) >= 2) and len(raw_matches) > 1:
            filtered = [m for m in raw_matches if m["doc_id"] != "SOP-LG-006"]
            if filtered:
                return filtered
        return raw_matches
