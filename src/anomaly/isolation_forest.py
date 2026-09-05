
import numpy as np
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_train_legit, contamination=0.02, random_state=42):
    """contamination should roughly match your expected fraud rate — it tells
    the forest what fraction of scored data it should expect to call anomalous."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_legit)
    return model

def anomaly_score(model, X):
    """Returns a score where HIGHER = more anomalous (note: sklearn's raw
    decision_function is inverted — lower/negative = more anomalous — so we
    flip the sign here to keep conventions consistent with the other models,
    where higher score always means higher risk)."""
    raw_scores = model.decision_function(X)
    return -raw_scores
