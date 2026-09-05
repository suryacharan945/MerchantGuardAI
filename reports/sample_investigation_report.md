# Incident Investigation Brief: b81a6b31-89b4-4077-b980-7e5d63df9c75

### 1. Executive Incident Overview
Transaction **b81a6b31-89b4-4077-b980-7e5d63df9c75** processed under merchant **m_07** was intercepted by MerchantGuard AI's hybrid ensemble with a calibrated risk score of **0.1023** (10.2% empirical fraud probability). The assessed transaction value is **INR 1,033.00**, representing an anomaly ratio of **4.50x** relative to the customer's historical 30-day baseline.

### 2. Forensic Signal Attribution
Automated SHAP feature attributions isolated the following primary indicators driving the anomaly score:
- **Amount Vs Avg Ratio** (attribution weight: +0.485)
- **Device Tx Count 1H** (attribution weight: +0.312)
- **Is Off Hours** (attribution weight: +0.144)

### 3. Regulatory & Policy Grounding
This pattern activates internal compliance protocol **[SOP-CT-002] High-Value Transaction Security SOP**. The protocol establishes that anomalous velocity and spending spikes require defensive operational intervention to mitigate chargeback risk and prevent unauthorized payment rails compromise.

### 4. Operational Disposition & Recommended Next Steps
- **Assigned Policy Action:** `BLOCK`
- **Automated Routing:** Aligned with SOP-CT-002 (Card Testing & Velocity Abuse Remediation) given calibrated risk.
- **Analyst Directives:**
  1. Freeze outgoing order fulfillment while transaction remains in `BLOCK` state.
  2. Verify device possession status and match customer registration records against IP telemetry.
  3. Log incident verdict into the merchant security portal audit ledger.
