import time
import hashlib
from datetime import datetime

class GatewayActionExecutor:
    """
    Simulated Razorpay Gateway API tool executor.
    Executes defensive actions safely with mock gateway response tokens.
    """
    def __init__(self):
        self.active_holds = {}
        self.blacklisted_devices = set()
        self.dispatched_otps = {}

    def execute_otp_challenge(self, tx_id: str, user_masked: str, amount: float) -> dict:
        otp_ref = f"otp_{hashlib.md5(f"{tx_id}:{time.time()}".encode()).hexdigest()[:8]}"
        self.dispatched_otps[tx_id] = {"otp_ref": otp_ref, "user": user_masked, "status": "PENDING_VERIFICATION"}
        return {
            "status": "SUCCESS",
            "action": "OTP_VERIFY",
            "tool_called": "gateway.challenge_2fa()",
            "details": f"OTP challenge {otp_ref} dispatched to {user_masked} for INR {amount:,.2f}.",
            "requires_human_followup": False
        }

    def execute_ledger_hold(self, tx_id: str, merchant_id: str, amount: float, reason: str) -> dict:
        hold_ticket = f"hld_{hashlib.md5(f"{tx_id}:{merchant_id}".encode()).hexdigest()[:8]}"
        self.active_holds[tx_id] = {"ticket": hold_ticket, "amount": amount, "timestamp": datetime.now().isoformat()}
        return {
            "status": "SUCCESS",
            "action": "HOLD",
            "tool_called": "settlement.place_escrow_hold()",
            "details": f"Escrow hold ticket {hold_ticket} registered. Settlement frozen for ops review.",
            "requires_human_followup": True
        }

    def execute_device_block(self, tx_id: str, device_masked: str, reason: str) -> dict:
        self.blacklisted_devices.add(device_masked)
        return {
            "status": "SUCCESS",
            "action": "BLOCK",
            "tool_called": "firewall.blacklist_device_token()",
            "details": f"Device token {device_masked} blacklisted on gateway edge. Transaction aborted.",
            "requires_human_followup": False
        }

    def execute_clearance(self, tx_id: str) -> dict:
        return {
            "status": "SUCCESS",
            "action": "ALLOW",
            "tool_called": "gateway.authorize_payment()",
            "details": f"Transaction {tx_id} cleared directly for immediate settlement.",
            "requires_human_followup": False
        }
