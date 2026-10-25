"""
Step 7: decide which hands go to a person.

    uv run python review.py          # also runs at the end of train.py

Three outcomes per hand, from its Rotten score:
    score <  t_low              pass to the supermarket
    t_low <= score < t_high     send to a person, who is assumed to decide correctly
    score >= t_high             reject

Compares four policies per 1,000 hands and per day: pass everything, a person
checks everything, the model alone (step 6 threshold), and model + human review
(the two thresholds that cost least). Works out reviews per day, hours and people.

Outputs:
  reports/review_policies.csv   the four policies side by side
  reports/review_summary.json   chosen thresholds, staffing, sensitivity
  reports/review_policies.png   cost per day of each policy
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from costing import BASE_RATE, CASES, COST_FN, COST_FP, REPORTS, THRESHOLDS, load_scores

# ---- Decision #9: every figure here is defended in REPORT.md ----
HANDS_PER_DAY = 50_000  # mid-sized Spanish ripening centre (18,000 t/year at ~1 kg per hand)
SECONDS_PER_CHECK = 5  # time for a person to look at one flagged hand
COST_PER_HOUR = 14.30  # warehouse inspector: ~EUR 11 gross + ~30% employer social security
SHIFT_HOURS = 8  # one person's working day
T_HIGH_CHOSEN = 0.80  # high line moved up from the cheapest (fitted to one test image) as a safety margin

ALL = 0.0  # a threshold every score passes: score >= 0 is always true
NONE = 1.01  # a threshold no score reaches


def outcome(df: pd.DataFrame, t_low: float, t_high: float, seconds: float = SECONDS_PER_CHECK,
            base_rate: float = BASE_RATE) -> dict:
    """Expected result of one (t_low, t_high) policy per 1,000 hands, plus daily staffing."""
    s, rotten = df.score.to_numpy(), df.defective.to_numpy(bool)
    reject = s >= t_high
    review = (s >= t_low) & ~reject
    passed = ~reject & ~review
    n_rotten, n_good = CASES * base_rate, CASES * (1 - base_rate)

    missed = n_rotten * passed[rotten].mean()
    false_alarms = n_good * reject[~rotten].mean()
    reviews = n_rotten * review[rotten].mean() + n_good * review[~rotten].mean()

    error_cost = missed * COST_FN + false_alarms * COST_FP
    labour_cost = reviews * COST_PER_HOUR * seconds / 3600
    per_day = HANDS_PER_DAY / CASES
    review_hours = reviews * per_day * seconds / 3600
    return {
        "t_low": t_low, "t_high": t_high,
        "missed_per_1000": missed, "false_alarms_per_1000": false_alarms,
        "reviews_per_1000": reviews, "review_share": reviews / CASES,
        "error_cost_per_1000": error_cost, "labour_cost_per_1000": labour_cost,
        "cost_per_1000": error_cost + labour_cost,
        "cost_per_day": (error_cost + labour_cost) * per_day,
        "reviews_per_day": reviews * per_day,
        "review_hours_per_day": review_hours,
        "people_needed": review_hours / SHIFT_HOURS,
    }


def best_band(df: pd.DataFrame, seconds: float = SECONDS_PER_CHECK) -> dict:
    """The (t_low, t_high) pair that costs least.

    Many pairs tie, because only a handful of test images fall between the
    thresholds. Among ties: first the fewest hands sent to a person, then the
    lowest t_low, i.e. the most cautious about passing a hand unchecked, since
    misses are the expensive error and production will be harder than the test set.
    """
    grid = [outcome(df, lo, hi, seconds) for lo in THRESHOLDS for hi in list(THRESHOLDS) + [NONE] if hi > lo]
    table = pd.DataFrame(grid)
    cheapest = table[table.cost_per_1000 <= table.cost_per_1000.min() + 0.01]
    return cheapest.sort_values(["reviews_per_1000", "t_low"], ascending=[True, True]).iloc[0].to_dict()


def main() -> None:
    df = load_scores(subset="full")
    step6 = json.loads((REPORTS / "cost_summary.json").read_text())["chosen_threshold"]
    band = best_band(df)
    chosen = outcome(df, band["t_low"], T_HIGH_CHOSEN)

    policies = {
        "Pass everything (no inspection)": outcome(df, NONE, NONE),
        "Person checks every hand": outcome(df, ALL, NONE),
        f"Model alone (reject at {step6:.2f})": outcome(df, step6, step6),
        f"Model + review, cheapest ({band['t_low']:.2f}-{band['t_high']:.2f})": band,
        f"Model + review, CHOSEN ({chosen['t_low']:.2f}-{chosen['t_high']:.2f})": chosen,
    }
    table = pd.DataFrame(policies).T

    slow = outcome(df, chosen["t_low"], chosen["t_high"], seconds=10)
    everyone_slow = outcome(df, ALL, NONE, seconds=10)
    summary = {
        "inputs": {"hands_per_day": HANDS_PER_DAY, "seconds_per_check": SECONDS_PER_CHECK,
                   "cost_per_hour": COST_PER_HOUR, "shift_hours": SHIFT_HOURS,
                   "cost_per_check": round(COST_PER_HOUR * SECONDS_PER_CHECK / 3600, 4),
                   "reviewer_assumed_correct": True, "base_rate": BASE_RATE,
                   "cost_fp": COST_FP, "cost_fn": COST_FN},
        "cheapest_band": {"t_low": band["t_low"], "t_high": band["t_high"]},
        "chosen_band": {"t_low": chosen["t_low"], "t_high": chosen["t_high"],
                        "why": "high line raised above the cheapest value, which was fitted to a single test image"},
        "policies": {k: {m: round(float(v), 4) for m, v in row.items()} for k, row in policies.items()},
        "sensitivity_10_seconds": {
            "model_plus_review_people": round(slow["people_needed"], 3),
            "model_plus_review_cost_per_day": round(slow["cost_per_day"], 2),
            "person_checks_everything_people": round(everyone_slow["people_needed"], 2),
            "person_checks_everything_cost_per_day": round(everyone_slow["cost_per_day"], 2),
        },
    }
    table.to_csv(REPORTS / "review_policies.csv")
    (REPORTS / "review_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(list(policies), table.cost_per_day.astype(float), color=["grey", "tab:orange", "tab:blue", "tab:olive", "tab:green"])
    for i, v in enumerate(table.cost_per_day.astype(float)):
        ax.text(v, i, f"  EUR {v:,.0f}/day", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Expected cost per day (EUR), {HANDS_PER_DAY:,} hands, base rate {BASE_RATE:.0%}")
    ax.set_title("Cost of each inspection policy")
    ax.set_xlim(0, table.cost_per_day.astype(float).max() * 1.35)
    fig.tight_layout()
    fig.savefig(REPORTS / "review_policies.png", dpi=150)

    show = table[["missed_per_1000", "false_alarms_per_1000", "review_share", "reviews_per_day",
                  "review_hours_per_day", "people_needed", "cost_per_1000", "cost_per_day"]].astype(float)
    print(f"\n== Policies: {HANDS_PER_DAY:,} hands/day, {SECONDS_PER_CHECK} s per check, "
          f"EUR {COST_PER_HOUR}/h, base rate {BASE_RATE:.0%} ==")
    with pd.option_context("display.width", 160, "display.float_format", "{:,.2f}".format):
        print(show.to_string())
    print(f"\nCheapest band on the test set: {band['t_low']:.2f}-{band['t_high']:.2f}")
    print(f"Chosen band: pass below {chosen['t_low']:.2f}, person checks "
          f"{chosen['t_low']:.2f}-{chosen['t_high']:.2f}, reject from {chosen['t_high']:.2f}")
    print(f"At 10 s per check: model + review needs {slow['people_needed']:.2f} people; "
          f"a person checking everything needs {everyone_slow['people_needed']:.1f}")
    print("\nSaved reports/review_policies.csv, reports/review_summary.json, reports/review_policies.png")


if __name__ == "__main__":
    main()
