
import csv, random, uuid, datetime
from datetime import timedelta

MERCHANTS = ["m_" + str(i) for i in range(1, 21)]
USERS = ["u_" + str(i) for i in range(1, 2001)]
PAY_METHODS = ["card", "upi", "wallet", "netbanking"]

def base_amount(merchant):
    return random.choice([50, 100, 200, 500, 1000, 2500])

def simulate(days=14, out="sample_transactions.csv", seed=42):
    random.seed(seed)  # reproducibility — required for judges
    start = datetime.datetime(2026, 8, 20, 0, 0, 0)
    rows = []
    for minute in range(60 * 24 * days):
        t = start + timedelta(minutes=minute)
        for _ in range(random.choices([0, 1, 2, 3], weights=[80, 10, 6, 4])[0]):
            merchant = random.choice(MERCHANTS)
            user = random.choice(USERS)
            amt_base = base_amount(merchant)
            amount = max(1, int(random.gauss(amt_base, amt_base * 0.5)))
            method = random.choice(PAY_METHODS)
            device = "dev_" + str(random.randint(1, 5000))
            ip_country = random.choice(["IN", "US", "GB", "NG", "PK"])

            fraud = 0
            if random.random() < 0.001 and amount > 1000:
                fraud = 1  # high-amount fraud
            if random.random() < 0.0005 and method == "card":
                fraud = 1  # card fraud pattern

            rows.append({
                "tx_id": str(uuid.uuid4()),
                "timestamp": t.isoformat(),
                "merchant_id": merchant,
                "user_id": user,
                "amount": amount,
                "payment_method": method,
                "device_fingerprint": device,
                "ip_country": ip_country,
                "is_fraud": fraud
            })

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, "-", len(rows), "rows")

if __name__ == "__main__":
    simulate(days=14, out="sample_transactions.csv")
