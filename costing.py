"""
Step 6: put a cost on being wrong and choose the confidence threshold that costs least.

    uv run python costing.py          # also runs at the end of train.py

Reads reports/image_scores.csv from evaluate.py: one score per test image, the
highest Rotten confidence the model predicted. A hand is rejected when its
score is at or above the threshold.

The test set is about 50% rotten; production is assumed 3%. So we measure the
model's rates within each class on the test set (recall on rotten hands, false
alarm rate on good hands), which do not depend on the mix, and apply them to
the production base rate.

Outputs:
  reports/cost_sweep.csv      cost per 1,000 hands at every threshold
  reports/cost_curve.png      cost against threshold, chosen threshold marked
  reports/cost_summary.json   chosen threshold, baselines, sensitivity, bootstrap
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT

REPORTS = ROOT / "reports"

# ---- Decision #7: every figure here is defended in REPORT.md ----
COST_FP = 0.85  # good hand rejected: what the supermarket would have paid for ~1 kg
COST_FN = 7.50  # rotten hand shipped: 50% chance it is caught x EUR 15 box rejected
COST_TP = 0.0  # rotten hand rejected: the correct outcome
COST_TN = 0.0  # good hand passed: the correct outcome
BASE_RATE = 0.03  # share of hands that are rotten in normal operation
CASES = 1000  # costs are reported per 1,000 hands

SUBSET = "full"  # threshold chosen on the full test split (Decision #5, revised)
CHECK_SUBSET = "unseen"  # photos never seen in training: reported alongside as a leakage check
THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2)  # 0.00 rejects everything, 1.00 almost nothing
PASS_ALL = 1.01  # above any score: nothing is rejected (the "do nothing" policy)
BOOTSTRAP_ROUNDS = 1000


def load_scores(model: str = "finetuned", subset: str = SUBSET) -> pd.DataFrame:
    df = pd.read_csv(REPORTS / "image_scores.csv")
    df = df[df.model == model]
    return df[df.unseen] if subset == "unseen" else df


def rates(scores: np.ndarray, defective: np.ndarray, threshold: float) -> tuple[float, float]:
    """Recall (share of rotten hands rejected) and false-alarm rate (share of good hands rejected)."""
    rejected = scores >= threshold
    return rejected[defective].mean(), rejected[~defective].mean()


def expected_counts(recall: float, false_alarm: float, base_rate: float = BASE_RATE, n: int = CASES) -> dict:
    rotten, good = n * base_rate, n * (1 - base_rate)
    return {"TP": rotten * recall, "FN": rotten * (1 - recall),
            "FP": good * false_alarm, "TN": good * (1 - false_alarm)}


def cost_of(counts: dict, c_fp: float = COST_FP, c_fn: float = COST_FN) -> float:
    return counts["FP"] * c_fp + counts["FN"] * c_fn + counts["TP"] * COST_TP + counts["TN"] * COST_TN


def sweep(df: pd.DataFrame, base_rate: float = BASE_RATE, c_fp: float = COST_FP, c_fn: float = COST_FN) -> pd.DataFrame:
    scores, defective = df.score.to_numpy(), df.defective.to_numpy(bool)
    rows = []
    for t in list(THRESHOLDS) + [PASS_ALL]:
        recall, false_alarm = rates(scores, defective, t)
        counts = expected_counts(recall, false_alarm, base_rate)
        flagged = counts["TP"] + counts["FP"]
        rows.append({"threshold": t, "recall": recall, "false_alarm_rate": false_alarm,
                     **counts, "precision": counts["TP"] / flagged if flagged else 0.0,
                     "cost": cost_of(counts, c_fp, c_fn)})
    return pd.DataFrame(rows)


def best_row(table: pd.DataFrame) -> pd.Series:
    return table.loc[table.cost.idxmin()]


def baselines(base_rate: float = BASE_RATE, c_fp: float = COST_FP, c_fn: float = COST_FN) -> dict:
    """Cost per 1,000 hands of the policies the model has to beat."""
    return {
        "pass_everything (do nothing / trivial / COCO zero-shot)": CASES * base_rate * c_fn,
        "reject_everything": CASES * (1 - base_rate) * c_fp,
        "perfect_model": 0.0,
    }


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base_rate in (0.01, 0.03, 0.05, 0.10):
        for c_fn in (3.75, 7.50, 15.0):
            b = best_row(sweep(df, base_rate, COST_FP, c_fn))
            pass_all = CASES * base_rate * c_fn
            rows.append({"base_rate": base_rate, "cost_fn": c_fn, "best_threshold": b.threshold,
                         "min_cost": round(b.cost, 1), "pass_all_cost": round(pass_all, 1),
                         "saving_vs_pass_all": round(pass_all - b.cost, 1)})
    return pd.DataFrame(rows)


def stable_range(df: pd.DataFrame, threshold: float, param: str, grid: np.ndarray) -> list:
    """Range of one input over which the chosen threshold stays within EUR 1 per 1,000 of the best.

    This is the README's "the answer changes if X drops below Z": outside this
    range a different threshold would be cheaper.
    """
    ok = []
    for value in grid:
        kwargs = {"base_rate": BASE_RATE, "c_fp": COST_FP, "c_fn": COST_FN, param: float(value)}
        table = sweep(df, **kwargs)
        at_chosen = table.loc[np.isclose(table.threshold, threshold), "cost"].iloc[0]
        if at_chosen - table.cost.min() <= 1.0:
            ok.append(float(value))
    return [round(min(ok), 3), round(max(ok), 3)] if ok else [float("nan"), float("nan")]


def bootstrap(df: pd.DataFrame, threshold: float, seed: int = 0) -> dict:
    """Resample test images within each class to see how much the result could move."""
    rng = np.random.default_rng(seed)
    rotten, good = df[df.defective], df[~df.defective]
    costs, best_thresholds = [], []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = pd.concat([rotten.sample(len(rotten), replace=True, random_state=rng.integers(1e9)),
                            good.sample(len(good), replace=True, random_state=rng.integers(1e9))])
        r, f = rates(sample.score.to_numpy(), sample.defective.to_numpy(bool), threshold)
        costs.append(cost_of(expected_counts(r, f)))
        best_thresholds.append(best_row(sweep(sample)).threshold)
    return {"cost_at_chosen_95ci": [round(float(np.percentile(costs, q)), 1) for q in (2.5, 97.5)],
            "best_threshold_95ci": [round(float(np.percentile(best_thresholds, q)), 2) for q in (2.5, 97.5)]}


def plot(table: pd.DataFrame, other: pd.DataFrame, best: pd.Series, base: dict) -> None:
    curve = table[table.threshold <= 1.0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve.threshold, curve.cost, label=f"Finetuned model ({SUBSET} test)", lw=2)
    other_curve = other[other.threshold <= 1.0]
    ax.plot(other_curve.threshold, other_curve.cost, ls="--", alpha=0.6,
            label=f"Finetuned model ({CHECK_SUBSET} test: leakage check, few images)")
    pass_all = list(base.values())[0]
    ax.axhline(pass_all, color="grey", ls=":", label=f"Pass everything: EUR {pass_all:.0f}")
    ax.axvline(best.threshold, color="red", ls="-", alpha=0.7)
    ax.scatter([best.threshold], [best.cost], color="red", zorder=5,
               label=f"Chosen: t = {best.threshold:.2f}, EUR {best.cost:.0f}")
    ax.set_ylim(0, max(pass_all * 1.6, best.cost * 2))
    ax.set_xlabel("Confidence threshold (reject the hand if its Rotten score is at or above it)")
    ax.set_ylabel(f"Expected cost per {CASES:,} hands (EUR)")
    ax.set_title(f"Cost against threshold  |  FP EUR {COST_FP}, FN EUR {COST_FN}, base rate {BASE_RATE:.0%}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "cost_curve.png", dpi=150)


def main() -> None:
    df = load_scores()
    table = sweep(df)
    other = sweep(load_scores(subset=CHECK_SUBSET))
    best = best_row(table)
    base = baselines()
    pass_all = list(base.values())[0]

    sens = sensitivity(df)
    summary = {
        "inputs": {"cost_fp": COST_FP, "cost_fn": COST_FN, "cost_tp": COST_TP, "cost_tn": COST_TN,
                   "base_rate": BASE_RATE, "cases": CASES, "subset": SUBSET,
                   "test_images": len(df), "test_rotten": int(df.defective.sum())},
        "chosen_threshold": float(best.threshold),
        "at_chosen_threshold": {k: round(float(best[k]), 4) for k in
                                ("recall", "false_alarm_rate", "precision", "TP", "FN", "FP", "TN", "cost")},
        "same_threshold_on_other_subset": round(float(other.loc[other.threshold == best.threshold, "cost"].iloc[0]), 1),
        "baselines_per_1000": {k: round(v, 1) for k, v in base.items()},
        "saving_vs_pass_everything": round(pass_all - float(best.cost), 1),
        "theory_threshold_if_scores_were_probabilities": round(COST_FP / (COST_FP + COST_FN), 3),
        "threshold_stays_best_while": {
            "cost_fn_between": stable_range(df, float(best.threshold), "c_fn", np.arange(0.5, 30.01, 0.25)),
            "base_rate_between": stable_range(df, float(best.threshold), "base_rate", np.arange(0.005, 0.2001, 0.005)),
        },
        "bootstrap": bootstrap(df, float(best.threshold)),
        "sensitivity": sens.to_dict(orient="records"),
    }

    table.to_csv(REPORTS / "cost_sweep.csv", index=False)
    (REPORTS / "cost_summary.json").write_text(json.dumps(summary, indent=2))
    plot(table, other, best, base)

    a = summary["at_chosen_threshold"]
    print(f"\n== Cost per {CASES:,} hands ({SUBSET} test, {len(df)} images, base rate {BASE_RATE:.0%}) ==")
    print(f"Chosen threshold: {best.threshold:.2f}")
    print(f"  recall {a['recall']:.1%}, false-alarm rate {a['false_alarm_rate']:.1%}, precision {a['precision']:.1%}")
    print(f"  per 1,000 hands: {a['TP']:.1f} caught, {a['FN']:.1f} missed, {a['FP']:.1f} false alarms")
    print(f"  expected cost EUR {a['cost']:.1f}  (95% bootstrap range {summary['bootstrap']['cost_at_chosen_95ci']})")
    print(f"  best threshold across bootstrap samples: {summary['bootstrap']['best_threshold_95ci']}")
    print(f"  same threshold on the {CHECK_SUBSET} test ({len(load_scores(subset=CHECK_SUBSET))} images): "
          f"EUR {summary['same_threshold_on_other_subset']}")
    print("\nBaselines per 1,000 hands:")
    for k, v in summary["baselines_per_1000"].items():
        print(f"  {k}: EUR {v}")
    print(f"Saving vs passing everything: EUR {summary['saving_vs_pass_everything']} per 1,000 hands")
    print(f"\nIf scores were true probabilities, theory says reject above {summary['theory_threshold_if_scores_were_probabilities']}.")
    print("That differs from the swept threshold because model scores are not calibrated probabilities.")
    stay = summary["threshold_stays_best_while"]
    print(f"Threshold {best.threshold:.2f} stays the best choice (within EUR 1 per 1,000) while a miss costs "
          f"EUR {stay['cost_fn_between'][0]}-{stay['cost_fn_between'][1]} and the base rate is "
          f"{stay['base_rate_between'][0]:.1%}-{stay['base_rate_between'][1]:.1%}.")
    print("\n== Sensitivity: best threshold and cost per 1,000 hands ==")
    print(sens.to_string(index=False))
    print("\nSaved reports/cost_sweep.csv, reports/cost_curve.png, reports/cost_summary.json")


if __name__ == "__main__":
    main()
