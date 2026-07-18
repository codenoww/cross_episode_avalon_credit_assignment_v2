"""
Bootstrap confidence intervals for individual message effects.

Answers a question nothing else in the pipeline currently answers: is THIS
SPECIFIC message's predicted effect statistically stable, or could normal
sampling variation easily produce a very different (or zero, or flipped-
sign) result?

Method: resample the training dataset with replacement N times, refit the
regression each time, and recompute this specific message's predicted
effect under each refit. If the resulting distribution of effects doesn't
cross zero (e.g. a 95% interval entirely positive or entirely negative),
that's real evidence this specific prediction is trustworthy, not a fluke
of one particular dataset sample.

Reference: Efron, B., & Tibshirani, R. J. (1994). An Introduction to the
Bootstrap. Chapman & Hall/CRC.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import statsmodels.api as sm
from gate2 import _build_full_dataset

np.random.seed(42)

N_BOOTSTRAP = 500


def _fit_on_sample(df_sample):
    df_sample = df_sample.copy()
    df_sample["interaction"] = df_sample["credit_score"] * df_sample["behavioral_score"]
    X = df_sample[["credit_score", "agent_skill", "interaction"]]
    X = sm.add_constant(X)
    y = df_sample["outcome"]
    return sm.OLS(y, X).fit()


def bootstrap_effect_interval(credit_score, agent_skill, behavioral_score, n_bootstrap=N_BOOTSTRAP, confidence=0.95):
    """
    Returns (point_estimate, ci_low, ci_high, stable: bool)
    stable=True means the confidence interval does not cross zero.
    """
    df = _build_full_dataset()
    n = len(df)

    bootstrap_effects = []
    for _ in range(n_bootstrap):
        sample = df.sample(n=n, replace=True)
        model = _fit_on_sample(sample)
        coef_credit = model.params.get("credit_score", 0)
        coef_interaction = model.params.get("interaction", 0)
        effect = (coef_credit * credit_score) + (coef_interaction * credit_score * behavioral_score)
        bootstrap_effects.append(effect)

    bootstrap_effects = np.array(bootstrap_effects)
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_effects, 100 * (alpha / 2))
    ci_high = np.percentile(bootstrap_effects, 100 * (1 - alpha / 2))
    point_estimate = np.mean(bootstrap_effects)

    stable = (ci_low > 0) or (ci_high < 0)  # interval does not cross zero

    return point_estimate, ci_low, ci_high, stable


if __name__ == "__main__":
    from gate2 import get_message_row, get_agent_win_rate

    print("Testing bootstrap CI on a few real messages (500 resamples each, may take a moment)...")
    for mid in [838, 839, 840, 841, 842]:
        msg = get_message_row(mid)
        skill = get_agent_win_rate(msg["sender_id"])
        point, low, high, stable = bootstrap_effect_interval(
            msg["final_credit_score"], skill, msg["behavioral_score"]
        )
        print(f"  message {mid}: point={point:.4f}, 95% CI=[{low:.4f}, {high:.4f}], stable={stable}") 