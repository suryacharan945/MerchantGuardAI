
import shap
import numpy as np
import pandas as pd

def get_explainer(model):
    return shap.TreeExplainer(model)

def explain_transaction(explainer, tx_row_df, feature_cols, top_n=5):
    """Return top_n (feature, shap_value) pairs for a single transaction row."""
    shap_values = explainer.shap_values(tx_row_df)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    idxs = np.argsort(-np.abs(sv[0]))[:top_n]
    return [(feature_cols[i], float(sv[0][i])) for i in idxs]
