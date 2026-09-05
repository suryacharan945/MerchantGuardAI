
def severity_bucket(calibrated_score):
    """Map a Platt-calibrated ensemble probability [0,1] to a severity label.
    v2 fix: Critical threshold is now bounded (<=0.99) and anchored to the
    75th percentile of confirmed-fraud scores, rather than an unbounded
    offset that could (and did) exceed the maximum possible score of 1.0.
    - Low/Medium boundary = 95th percentile of LEGIT transaction scores
    - Medium/High boundary = F1-optimal decision threshold on calibrated scores
    - High/Critical boundary = 75th percentile of FRAUD scores, bounded to [medium_high+0.02, 0.99]
    """
    if calibrated_score < 0.071:
        return "Low"
    elif calibrated_score < 0.89:
        return "Medium"
    elif calibrated_score < 0.988:
        return "High"
    else:
        return "Critical"

SEVERITY_THRESHOLDS = {
    "low_medium": 0.071,
    "medium_high": 0.89,
    "high_critical": 0.988
}

def confidence_note(calibrated_score, n_test_fraud=28):
    return (f"Risk buckets derived from a {n_test_fraud}-fraud-case validation set "
            f"— treat boundary cases (within ~0.05 of a cutoff) as lower-confidence.")
