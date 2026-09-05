# Known Limitations — MerchantGuard AI

## Recall gap on low-signal, small-amount fraud

During testing, 7 of 28 fraud cases in the held-out test set (small-amount
refund-abuse and low-signal stolen-card patterns, ₹250–3,000 per transaction) were not
flagged by the detection pipeline. These cases scored below 0.05 on the calibrated
ensemble risk score and did not trigger any behavioral red flag (unrecognized IP,
off-hours timing, high device count) — they are, by every feature currently engineered,
close to indistinguishable from ordinary legitimate transactions.

### What we tried
1. **Amount-only safety net** (force minimum action above the 99th percentile of legit
   transaction amounts): added ₹7,703 in false-positive friction with zero increase in
   fraud caught, since every missed case had an amount below this threshold.
2. **Amount + behavioral-flag safety net** (moderate amount threshold combined with at
   least one corroborating flag): caught marginally more fraud (+₹1,400 prevented) but
   at a much larger false-positive cost (+₹12,083 friction) — net business impact was
   worse than having no safety net at all.

### Decision
Both policy-layer interventions were rejected after being tested against actual
business-impact numbers, not assumed to work. The original severity-based policy
(no amount override) has the highest net benefit of all three versions tested and is
what ships in this submission.

### Why this is a model problem, not a policy problem
No threshold on amount or existing behavioral features can separate these fraud cases
from legit ones, because the signal genuinely isn't present in the current feature set
at the scale these fraud cases operate. The correct fix is upstream: additional
training examples of this specific pattern, or a new engineered feature that captures
whatever distinguishes "quiet" low-amount fraud from normal small purchases — not a
downstream rule that necessarily sweeps in unrelated legitimate transactions to
compensate.

### Why we're disclosing this instead of hiding it
A safety-net rule that *looks* protective in a demo but reduces net benefit under
actual measurement is worse than no rule — it would misrepresent the system's real
performance. We chose to test our own fix against the actual numbers rather than
assume it worked, and revert it when the numbers said no.


## Calibration overconfidence in the highest-risk score range (CORRECTED)

Binomial testing (Step 8l) confirmed the [0.5, 1.0) calibrated-score range shows
statistically significant overconfidence (p<0.0001): Platt scaling predicts ~83% mean
fraud probability in this range, but the observed test-set rate is ~44%.

**Correction to an earlier, unverified explanation:** we initially guessed this was due
to sparse training data at the extreme tail limiting Platt scaling's ability to fit that
region. Direct verification showed this guess was wrong — 227 training examples
(well above a sparsity threshold) fall in this range, and critically, the TRAINING data
itself already shows only a ~49% fraud rate there, closely
matching the ~44% observed in test. **The real cause is that Platt scaling (a single
logistic sigmoid fit on one score feature) is structurally too simple to capture the true,
more complex relationship between ensemble score and fraud probability at the extreme end
of the range** — not a data-sparsity problem, since adequate data was available.

**Practical implication (unchanged):** treat high calibrated scores as a strong relative
risk ranking, not a literal probability — a 0.9 score means "meaningfully riskier than a
0.5 score," not "90% likely to be fraud." A more flexible calibration method (e.g.
isotonic regression with more training data, or binning-based calibration) could reduce
this specific gap in future iterations, at the cost of the plateau/tie issues isotonic
showed at this dataset's current size (see Step 7b).
