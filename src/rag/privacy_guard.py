import hashlib

def mask_user_id(user_id: str) -> str:
    """Anonymize user ID via SHA-256 truncated salt."""
    if not user_id:
        return "usr_anon"
    return f"usr_{hashlib.sha256(str(user_id).encode()).hexdigest()[:8]}"

def mask_card_bin(card_bin: str) -> str:
    """Mask card BIN to prevent raw credential exposure."""
    bin_str = str(card_bin)
    if len(bin_str) >= 6:
        return f"{bin_str[:4]}XX"
    return "XXXX"

def mask_device(device_id: str) -> str:
    """Preserve device category/prefix while hashing specific identifiers."""
    dev_str = str(device_id)
    if dev_str.startswith("dev_testing_"):
        return f"dev_testing_{hashlib.sha256(dev_str.encode()).hexdigest()[:6]}"
    return f"dev_{hashlib.sha256(dev_str.encode()).hexdigest()[:6]}"

def sanitize_transaction_payload(tx_dict: dict) -> dict:
    """Return a privacy-safe dictionary suitable for LLM context."""
    sanitized = dict(tx_dict)
    if "user_id" in sanitized:
        sanitized["user_id"] = mask_user_id(sanitized["user_id"])
    if "card_bin" in sanitized:
        sanitized["card_bin"] = mask_card_bin(sanitized["card_bin"])
    if "device_fingerprint" in sanitized:
        sanitized["device_fingerprint"] = mask_device(sanitized["device_fingerprint"])
    return sanitized
