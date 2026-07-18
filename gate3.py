"""
Gate 3 -- Refutation tests

Updated to work with the new one-fit-per-dataset Gate 2 approach.
Since we no longer refit a fresh DoWhy model per message, refutation here
works differently: we test whether the FITTED MODEL ITSELF is robust
(placebo: does a randomly permuted treatment column still fit similarly
well? random common cause: does adding a fake confound change the fitted
coefficients much?) -- tested ONCE against the whole dataset, then applied
to every message's individual effect via the same fitted coefficients.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from gate2 import get_message_row, get_agent_win_rate, _build_full_dataset, predict_individual_effect

np.random.seed(42) 

_placebo_check_cache = None
_random_common_cause_check_cache = None


def _run_placebo_check():
    """
    Permutes credit_score across rows, refits, and checks whether the
    permuted model's coefficients collapse toward zero (expected if the
    real relationship is genuine, not coincidental).
    """
    global _placebo_check_cache
    if _placebo_check_cache is not None:
        return _placebo_check_cache

    df = _build_full_dataset().copy()
    df["interaction"] = df["credit_score"] * df["behavioral_score"]
    X = df[["credit_score", "agent_skill", "interaction"]]
    X = sm.add_constant(X)
    y = df["outcome"]
    real_model = sm.OLS(y, X).fit()
    real_coef = real_model.params.get("credit_score", 0)

    df_permuted = df.copy()
    df_permuted["credit_score"] = np.random.permutation(df_permuted["credit_score"].values)
    df_permuted["interaction"] = df_permuted["credit_score"] * df_permuted["behavioral_score"]
    X_p = df_permuted[["credit_score", "agent_skill", "interaction"]]
    X_p = sm.add_constant(X_p)
    placebo_model = sm.OLS(y, X_p).fit()
    placebo_coef = placebo_model.params.get("credit_score", 0)

    passed = abs(real_coef) > abs(placebo_coef) * 1.5
    _placebo_check_cache = {"real_coef": real_coef, "placebo_coef": placebo_coef, "passed": passed}
    return _placebo_check_cache


def _run_random_common_cause_check():
    """
    Adds a fake random confound column, refits, and checks whether the
    credit_score coefficient stays roughly stable (expected if the real
    relationship isn't fragile/spurious).
    """
    global _random_common_cause_check_cache
    if _random_common_cause_check_cache is not None:
        return _random_common_cause_check_cache

    df = _build_full_dataset().copy()
    df["interaction"] = df["credit_score"] * df["behavioral_score"]
    X = df[["credit_score", "agent_skill", "interaction"]]
    X = sm.add_constant(X)
    y = df["outcome"]
    real_model = sm.OLS(y, X).fit()
    real_coef = real_model.params.get("credit_score", 0)

    df_rcc = df.copy()
    df_rcc["fake_confound"] = np.random.normal(0, 1, size=len(df_rcc))
    X_rcc = df_rcc[["credit_score", "agent_skill", "interaction", "fake_confound"]]
    X_rcc = sm.add_constant(X_rcc)
    rcc_model = sm.OLS(y, X_rcc).fit()
    rcc_coef = rcc_model.params.get("credit_score", 0)

    if real_coef != 0:
        pct_change = abs(rcc_coef - real_coef) / abs(real_coef)
        passed = pct_change < 0.5
    else:
        passed = None

    _random_common_cause_check_cache = {"real_coef": real_coef, "rcc_coef": rcc_coef, "passed": passed}
    return _random_common_cause_check_cache


def run_gate3_for_message(message_id, path="option_a"):
    """
    Returns this message's individual effect (from Gate 2) plus the
    dataset-level refutation results (computed once, shared across all
    messages, since they test the fitted model's overall robustness).
    """
    msg = get_message_row(message_id)
    skill = get_agent_win_rate(msg["sender_id"])

    effect = predict_individual_effect(
        credit_score=msg["final_credit_score"],
        agent_skill=skill,
        behavioral_score=msg["behavioral_score"],
    )

    df = _build_full_dataset()
    placebo = _run_placebo_check()
    rcc = _run_random_common_cause_check()

    return {
        "message_id": message_id,
        "path": path,
        "original_effect": effect,
        "n_rows": len(df),
        "reliable": len(df) >= 8,
        "placebo_new_effect": placebo["placebo_coef"],
        "placebo_passed": placebo["passed"],
        "random_common_cause_new_effect": rcc["rcc_coef"],
        "random_common_cause_passed": rcc["passed"],
    }


if __name__ == "__main__":
    print("Testing Gate 3 on a few different real messages...")
    for mid in [122, 127, 133]:
        result = run_gate3_for_message(mid)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print() 