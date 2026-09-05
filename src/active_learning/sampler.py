
import numpy as np

def uncertainty_sample(model, X_pool, n_samples=20, strategy="entropy"):
    """Select the n_samples the model is LEAST confident about — i.e.
    predictions closest to 0.5 (maximum entropy for a binary classifier).
    These are the cases most valuable for a human to label next, since the
    model genuinely doesn't know which way they go (as opposed to easy
    cases it already handles confidently)."""
    scores = model.predict(X_pool, num_iteration=model.best_iteration)

    if strategy == "entropy":
        # Distance from the maximally-uncertain point (0.5) — smaller = more uncertain
        uncertainty = -np.abs(scores - 0.5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Most uncertain first
    order = np.argsort(uncertainty)[::-1][:n_samples]
    return order, scores[order]

def simulate_labeling(X_pool, y_pool_true, selected_indices):
    """Simulates a human analyst labeling the selected uncertain cases.
    In production this would be a UI queue; here we reveal the true labels
    for the selected subset only, mimicking analyst review."""
    return y_pool_true.iloc[selected_indices].reset_index(drop=True)
