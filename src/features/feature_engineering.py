
import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_off_hours"] = df["hour"].apply(lambda h: 1 if (h < 6 or h > 22) else 0)

    # User-level velocity features (rolling, per user)
    df["prev_tx_count_24h"] = (
        df.groupby("user_id")["timestamp"]
        .transform(lambda s: s.rolling("24h", on=s).count() if False else None)
    )
    # Simpler, robust rolling implementation (avoids index alignment issues):
    counts = []
    avg_amounts = []
    time_since_last = []
    device_nunique = []
    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        ts = grp["timestamp"]
        amt = grp["amount"]
        cnt_24h, avg_30d, tsince, dev_nuniq = [], [], [], []
        seen_devices = set()
        for i in range(len(grp)):
            window_24h = ts[(ts <= ts.iloc[i]) & (ts > ts.iloc[i] - pd.Timedelta(hours=24))]
            cnt_24h.append(len(window_24h) - 1)
            window_30d = amt[(ts <= ts.iloc[i]) & (ts > ts.iloc[i] - pd.Timedelta(days=30))]
            avg_30d.append(window_30d.mean() if len(window_30d) else amt.iloc[i])
            tsince.append((ts.iloc[i] - ts.iloc[i-1]).total_seconds() / 60 if i > 0 else 9999)
            seen_devices.add(grp["device_fingerprint"].iloc[i])
            dev_nuniq.append(len(seen_devices))
        counts.extend(cnt_24h); avg_amounts.extend(avg_30d)
        time_since_last.extend(tsince); device_nunique.extend(dev_nuniq)

    df["prev_tx_count_24h"] = counts
    df["avg_amount_30d"] = avg_amounts
    df["time_since_last_tx_min"] = time_since_last
    df["user_device_nunique"] = device_nunique
    df["amount_vs_avg_ratio"] = df["amount"] / df["avg_amount_30d"].replace(0, 1)

    # IP mismatch: does this tx use a different country than user\'s most common one
    home_country = df.groupby("user_id")["ip_country"].agg(lambda s: s.mode()[0])
    df["home_country"] = df["user_id"].map(home_country)
    df["ip_country_mismatch"] = (df["ip_country"] != df["home_country"]).astype(int)

    return df

def encode_categoricals(df, cols):
    for c in cols:
        df[c + "_code"] = df[c].astype("category").cat.codes
    return df


def add_velocity_features(df):
    """Device- and merchant-level burst detection — catches card-testing patterns
    that user-level features (prev_tx_count_24h) structurally miss, since
    card-testing uses ONE device against MANY different users."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]

    device_counts, merchant_counts = [], []
    for i in range(len(df)):
        t_now = ts.iloc[i]
        window_start = t_now - pd.Timedelta(hours=1)

        dev = df["device_fingerprint"].iloc[i]
        dev_mask = (df["device_fingerprint"] == dev) & (ts <= t_now) & (ts > window_start)
        device_counts.append(int(dev_mask.sum()) - 1)

        merch = df["merchant_id"].iloc[i]
        merch_mask = (df["merchant_id"] == merch) & (ts <= t_now) & (ts > window_start)
        merchant_counts.append(int(merch_mask.sum()) - 1)

    df["device_tx_count_1h"] = device_counts
    df["merchant_tx_count_1h"] = merchant_counts
    return df
