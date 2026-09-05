
import csv, random, uuid, datetime
from datetime import timedelta

MERCHANTS = ["m_" + str(i) for i in range(1, 21)]
USERS = ["u_" + str(i) for i in range(1, 2001)]
PAY_METHODS = ["card", "upi", "wallet", "netbanking"]
PAY_WEIGHTS = [40, 35, 15, 10]
IP_COUNTRIES = ["IN", "US", "GB", "NG", "PK"]
IP_WEIGHTS = [85, 5, 4, 3, 3]
MERCHANT_CATEGORIES = ["retail", "electronics", "travel", "food", "digital_goods"]

def shared_country():
    # SAME distribution used for legit AND fraud rows — no shortcut signal here
    return random.choices(IP_COUNTRIES, weights=IP_WEIGHTS)[0]

def shared_method():
    return random.choices(PAY_METHODS, weights=PAY_WEIGHTS)[0]

def base_amount(merchant):
    return random.choice([50, 100, 200, 500, 1000, 2500])

def make_row(merchant, user, amount, method, device, ip_country, ts, is_fraud, chargeback=0):
    return {
        "tx_id": str(uuid.uuid4()), "timestamp": ts.isoformat(),
        "merchant_id": merchant, "user_id": user, "amount": amount, "currency": "INR",
        "payment_method": method, "device_fingerprint": device, "ip_country": ip_country,
        "card_bin": str(random.randint(400000, 499999)),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "response_code": "00" if not is_fraud else random.choice(["00", "05", "51"]),
        "chargeback_flag": chargeback, "is_fraud": is_fraud
    }

def simulate(days=14, out="sample_transactions_v3.csv", seed=42, target_fraud_rate=0.02):
    random.seed(seed)
    start = datetime.datetime(2026, 8, 20, 0, 0, 0)
    rows = []

    for minute in range(60 * 24 * days):
        t = start + timedelta(minutes=minute)
        for _ in range(random.choices([0, 1, 2, 3], weights=[80, 10, 6, 4])[0]):
            merchant = random.choice(MERCHANTS)
            user = random.choice(USERS)
            amt_base = base_amount(merchant)
            amount = max(1, int(random.gauss(amt_base, amt_base * 0.3)))
            rows.append(make_row(merchant, user, amount, shared_method(),
                                  "dev_" + str(random.randint(1, 5000)),
                                  shared_country(), t, 0))

    n_legit = len(rows)
    n_fraud_target = int(n_legit * target_fraud_rate / (1 - target_fraud_rate))
    per_type = n_fraud_target // 4

    # Type 1: High-amount — amount is the ONLY signal (country/method match background)
    for _ in range(per_type):
        merchant = random.choice(MERCHANTS)
        user = random.choice(USERS)
        amount = base_amount(merchant) * random.randint(10, 20)
        t = start + timedelta(minutes=random.randint(0, 60*24*days - 1))
        rows.append(make_row(merchant, user, amount, shared_method(),
                              "dev_" + str(random.randint(1, 5000)),
                              shared_country(), t, 1))

    # Type 2: Card-testing — signal is VELOCITY (many tx, same device, tight burst)
    for _ in range(per_type // 5):
        merchant = random.choice(MERCHANTS)
        device = "dev_testing_" + str(uuid.uuid4())[:8]
        t0 = start + timedelta(minutes=random.randint(0, 60*24*days - 30))
        for k in range(6):
            user = random.choice(USERS)
            amount = random.randint(1, 20)
            t = t0 + timedelta(minutes=k)
            rows.append(make_row(merchant, user, amount, "card", device,
                                  shared_country(), t, 1))

    # Type 3: Stolen-card — signal is DEVICE NOVELTY (one user, many new devices, fast)
    for _ in range(per_type // 5):
        user = random.choice(USERS)
        t0 = start + timedelta(minutes=random.randint(0, 60*24*days - 30))
        for k in range(6):
            merchant = random.choice(MERCHANTS)
            amount = base_amount(merchant) * random.randint(2, 5)
            device = "dev_" + str(uuid.uuid4())[:8]  # always new device
            t = t0 + timedelta(minutes=k * 3)
            rows.append(make_row(merchant, user, amount, shared_method(), device,
                                  shared_country(), t, 1))

    # Type 4: Refund abuse — signal is AMOUNT RATIO + off-hours + chargeback
    for _ in range(per_type):
        merchant = random.choice(MERCHANTS)
        user = random.choice(USERS)
        amount = base_amount(merchant) * random.randint(3, 8)
        device = "dev_" + str(random.randint(1, 5000))
        hour_offset = random.choice([1, 2, 3, 23])  # bias toward off-hours
        t = (start + timedelta(minutes=random.randint(0, 60*24*days - 1))).replace(hour=hour_offset)
        rows.append(make_row(merchant, user, amount, shared_method(), device,
                              shared_country(), t, 1, chargeback=1))

    random.shuffle(rows)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} - {len(rows)} rows, {sum(r['is_fraud'] for r in rows)} fraud rows")

if __name__ == "__main__":
    simulate(days=14, out="sample_transactions_v3.csv")
