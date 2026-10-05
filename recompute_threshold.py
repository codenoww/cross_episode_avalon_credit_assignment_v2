"""
recompute_threshold.py -- recomputes Gate 2's confirmation threshold.

    tau = 0.5 * sigma(ITE)        (Cohen's 1988 "medium" effect-size convention)

where ITE(m) = beta_c * c(m) + beta_i * c(m) * d(m) is each training message's
individual treatment effect under the single regression Gate 2 fits
(outcome ~ credit + agent_skill + credit x behavioural_delta), and sigma is its
standard deviation over the training set.

This script was auto-generated once and later lost; threshold_config.py still holds
the value from an early 662-message snapshot (0.02528), which is stale. Running this
reproduces the paper's figure for the original 50-game tournament and shows the value
for the current database.

Usage:
    python recompute_threshold.py                    # print for both scopes, change nothing
    python recompute_threshold.py --write all        # write threshold_config.py from ALL games
    python recompute_threshold.py --write original   # write it from the original 50 games only

WARNING: writing a new threshold changes which candidate edges are confirmed the next
time the graph is built. Existing edges in the database are not recomputed.
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import psycopg2
import statsmodels.api as sm
from dotenv import load_dotenv

import gate2
from data_paths import DATA_PATH

load_dotenv()

CONFIG_PATH = "threshold_config.py"
ORIGINAL_PREFIX = "avalon_2025"    # the original 50-game tournament's episode ids


def episode_of_message():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "127.0.0.1"), port=os.environ.get("DB_PORT", "5434"),
        dbname=os.environ.get("DB_NAME", "avalon_research"), user=os.environ.get("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
    )
    cur = conn.cursor()
    cur.execute("SELECT id, episode_id FROM messages")
    out = dict(cur.fetchall())
    conn.close()
    return out


def sigma_ite(df):
    d = df.copy()
    d["interaction"] = d["credit_score"] * d["behavioral_score"]
    model = sm.OLS(d["outcome"], sm.add_constant(d[["credit_score", "agent_skill", "interaction"]])).fit()
    ite = (model.params["credit_score"] * d["credit_score"]
           + model.params["interaction"] * d["credit_score"] * d["behavioral_score"])
    return len(d), float(ite.std()), float(model.params["credit_score"])


def write_config(n, sigma, scope):
    tau = round(0.5 * sigma, 5)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(f'"""\nSingle source of truth for Gate 2\'s confirmation threshold.\n'
                f"Written by recompute_threshold.py -- Cohen's (1988) MEDIUM effect-size convention\n"
                f"(0.5 SD), computed on scope='{scope}': n={n} messages, std dev={sigma:.4f}.\n"
                f"Reference: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences\n"
                f"(2nd ed.). Lawrence Erlbaum Associates.\n\"\"\"\n\n")
        f.write(f"GATE2_CONFIRMATION_THRESHOLD = {tau}\n")
    print(f"wrote {CONFIG_PATH}: {tau}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", choices=["all", "original"], help="write threshold_config.py from this scope")
    args = ap.parse_args()

    df = gate2._build_full_dataset()
    ep = episode_of_message()
    df = df.assign(ep=df["message_id"].map(ep))
    scopes = {"all": df, "original": df[df["ep"].astype(str).str.startswith(ORIGINAL_PREFIX)]}

    try:
        current = float(open(CONFIG_PATH).read().split("=")[-1])
    except Exception:
        current = None
    print(f"current threshold_config.py: {current}")
    results = {}
    for name, d in scopes.items():
        n, s, b = sigma_ite(d)
        results[name] = (n, s)
        print(f"scope={name:<8} n={n:<5} sigma(ITE)={s:.4f}  tau=0.5*sigma={0.5 * s:.5f}  credit coefficient={b:.4f}")
    if args.write:
        n, s = results[args.write]
        write_config(n, s, args.write)


if __name__ == "__main__":
    main()
