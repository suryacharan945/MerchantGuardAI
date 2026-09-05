import os
import json
import time
from src.agent.action_tools import GatewayActionExecutor
from src.agent.evidence_builder import build_evidence_package
from src.agent.agent_policy import AutonomousAgentPolicy

class MerchantGuardAgentOrchestrator:
    def __init__(self, project_root, auto_mode=True):
        self.project_root = project_root
        self.auto_mode = auto_mode
        self.executor = GatewayActionExecutor()
        self.policy = AutonomousAgentPolicy()
        self.active_learning_queue = []

    def process_transaction(self, tx_dict: dict, score: float, shap_cues: list, similar_cases: list, sops: list) -> dict:
        t0 = time.perf_counter()

        # 1. Evidence Assembly
        evidence = build_evidence_package(tx_dict, score, shap_cues, similar_cases, sops)

        # 2. Decision Formulation
        decision = self.policy.evaluate_triage(evidence)
        action = decision["recommended_action"]

        # 3. Action Execution / Triage Routing
        executed = False
        execution_report = {}

        if self.auto_mode and decision["can_auto_execute"]:
            if action == "BLOCK":
                execution_report = self.executor.execute_device_block(tx_dict.get("tx_id"), "dev_token", decision["rationale"])
            elif action == "HOLD":
                execution_report = self.executor.execute_ledger_hold(tx_dict.get("tx_id"), tx_dict.get("merchant_id"), tx_dict.get("amount", 0), decision["rationale"])
            elif action == "OTP_VERIFY":
                execution_report = self.executor.execute_otp_challenge(tx_dict.get("tx_id"), "usr_masked", tx_dict.get("amount", 0))
            else:
                execution_report = self.executor.execute_clearance(tx_dict.get("tx_id"))
            executed = True
        else:
            execution_report = {
                "status": "ESCALATED_TO_HUMAN",
                "action": action,
                "tool_called": "triage.route_to_ops_console()",
                "details": f"Low confidence ({decision['confidence']}) or safety breaker triggered. Awaiting human approval.",
                "requires_human_followup": True
            }
            # Enqueue low confidence cases for Active Learning
            if decision["confidence"] in ["low", "medium"]:
                self.active_learning_queue.append({
                    "tx_id": tx_dict.get("tx_id"),
                    "score": score,
                    "prescribed_action": action,
                    "confidence": decision["confidence"]
                })

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "tx_id": tx_dict.get("tx_id"),
            "calibrated_score": score,
            "action": action,
            "confidence": decision["confidence"],
            "auto_executed": executed,
            "tool_details": execution_report["details"],
            "rationale": decision["rationale"],
            "latency_ms": round(latency_ms, 2)
        }
