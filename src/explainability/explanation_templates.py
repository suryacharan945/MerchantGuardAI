
def to_plain_english(shap_top, tx_row):
    """Convert top SHAP features into a merchant-friendly explanation."""
    lines = []
    for feat, val in shap_top:
        if val <= 0:
            continue  # only explain features pushing TOWARD fraud
        if feat == "amount_vs_avg_ratio":
            ratio = tx_row.get("amount_vs_avg_ratio", None)
            if ratio is not None:
                lines.append(f"Amount is {ratio:.1f}x this user's typical spend.")
        elif feat == "user_device_nunique":
            lines.append("Multiple distinct devices used by this user recently.")
        elif feat == "ip_country_mismatch":
            lines.append("Transaction originates from an unrecognized country for this user.")
        elif feat == "prev_tx_count_24h":
            lines.append(f"Unusually high transaction frequency in the last 24 hours.")
        elif feat == "is_off_hours":
            lines.append("Transaction occurred during off-hours.")
        elif feat == "ip_country_code":
            lines.append("Transaction country differs from typical merchant traffic.")
        elif feat == "payment_method_code":
            lines.append("Payment method pattern is atypical for this context.")
        else:
            lines.append(f"{feat.replace('_', ' ')} contributed to the risk score.")
    if not lines:
        lines = ["Multiple minor factors combined to raise the risk score."]
    return " ".join(lines[:3])  # cap at 3 sentences, merchant-friendly length
