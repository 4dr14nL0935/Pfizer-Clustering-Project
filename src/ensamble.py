"""
optimize_thresholds.py
======================
Standalone script — does NOT touch the dashboard HTML.

Goal: find decision thresholds (thr_A, thr_C) that minimize a business-cost
function, and compare against the current heuristic (thr_A=0.70, thr_C=0.30).

THE COST MATRIX (rows = true segment, columns = predicted segment)

                    pred_A    pred_B    pred_C
    true_A          0.0       0.3       1.0      <- A->C is COSTLY (waste sales force)
    true_B          0.3       0.0       0.3
    true_C          5.0       0.3       0.0      <- C->A is CATASTROPHIC (lose prescriber)

The 5:1 ratio between catastrophic and costly errors comes from the project's
own report: every SEG_C lost as SEG_A is recurring revenue lost (the doctor
keeps prescribing competitor brands, unreached); every SEG_A visited as SEG_C
is a one-time cost that the rep can de-prioritize after a few visits.

This script also runs a SENSITIVITY ANALYSIS: it re-runs the grid search at
multiple cost ratios (3:1, 5:1, 7:1, 10:1) so you can show the robustness of
the optimum.

USAGE
    python optimize_thresholds.py --input doctors_aggregated.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost not installed.  Run: pip install xgboost")


VALID_LABELS = ("SEG_A", "SEG_B", "SEG_C")

# Same hyperparams as the chosen ordinal model
ORDINAL_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
SEG_C_SAMPLE_WEIGHT = 2.0

# Current heuristic thresholds (the baseline we want to beat)
HEURISTIC_THR_A = 0.70
HEURISTIC_THR_C = 0.30


# ============================ Ordinal model ============================
def fit_ordinal(X_train, y_train):
    y_geq_b = (y_train != "SEG_A").astype(int)
    y_geq_c = (y_train == "SEG_C").astype(int)
    m1 = xgb.XGBClassifier(**ORDINAL_PARAMS); m1.fit(X_train, y_geq_b)
    sw = np.where(y_geq_c == 1, SEG_C_SAMPLE_WEIGHT, 1.0)
    m2 = xgb.XGBClassifier(**ORDINAL_PARAMS); m2.fit(X_train, y_geq_c, sample_weight=sw)
    return m1, m2


def predict_proba_ordinal(m1, m2, X):
    """Returns ordinal probabilities WITHOUT applying any threshold."""
    p_b = m1.predict_proba(X)[:, 1]
    p_c = np.minimum(m2.predict_proba(X)[:, 1], p_b)
    P_A = 1 - p_b
    P_B = p_b - p_c
    P_C = p_c
    s = P_A + P_B + P_C
    return P_A / s, P_B / s, P_C / s


def oof_probabilities(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    P_A = np.zeros(len(y)); P_B = np.zeros(len(y)); P_C = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{n_splits}...", flush=True)
        m1, m2 = fit_ordinal(X[tr], y[tr])
        P_A[va], P_B[va], P_C[va] = predict_proba_ordinal(m1, m2, X[va])
    return P_A, P_B, P_C


def add_ratios(df):
    eps = 1e-6
    out = df.copy()
    pairs = [
        ("ratio_UC_over_ORAL",    "UC_TRX_sum",   "ORAL_TRX_sum"),
        ("ratio_UC_over_IL23",    "UC_TRX_sum",   "IL23_TRX_sum"),
        ("ratio_ORAL_over_IL23",  "ORAL_TRX_sum", "IL23_TRX_sum"),
        ("ratio_UC_NRX_TRX",      "UC_NRX_sum",   "UC_TRX_sum"),
        ("ratio_DETAILS_per_TRX", "DETAILS_sum",  "UC_TRX_sum"),
    ]
    for new, num, den in pairs:
        if num in out.columns and den in out.columns:
            out[new] = out[num] / (out[den] + eps)
    bcols = [c for c in out.columns if c.startswith("BRAND") and "_TRX_sum" in c]
    if bcols:
        out["brand_diversity"] = (out[bcols] > 0).sum(axis=1)
    return out


# ============================ Cost & evaluation ============================
def build_cost_matrix(c_to_a=5.0, a_to_c=1.0, b_error=0.3):
    return {
        "SEG_A": {"SEG_A": 0.0, "SEG_B": b_error, "SEG_C": a_to_c},
        "SEG_B": {"SEG_A": b_error, "SEG_B": 0.0, "SEG_C": b_error},
        "SEG_C": {"SEG_A": c_to_a, "SEG_B": b_error, "SEG_C": 0.0},
    }


def total_cost(y_true, y_pred, cost_matrix):
    """Vectorized total cost over all HCPs."""
    cost = 0.0
    for t in VALID_LABELS:
        for p in VALID_LABELS:
            mask = (y_true == t) & (y_pred == p)
            cost += mask.sum() * cost_matrix[t][p]
    return float(cost)


def evaluate(y_true, P_A, P_C, thr_a, thr_c, cost_matrix):
    pred = np.where(P_A >= thr_a, "SEG_A",
           np.where(P_C >= thr_c, "SEG_C", "SEG_B"))
    cost = total_cost(y_true, pred, cost_matrix)
    acc = float((pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    mask_C = y_true == "SEG_C"
    n_C = int(mask_C.sum())
    n_C_correct = int((pred[mask_C] == "SEG_C").sum())
    n_C_lost_as_A = int(((y_true == "SEG_C") & (pred == "SEG_A")).sum())
    n_A = int((y_true == "SEG_A").sum())
    n_A_to_C = int(((y_true == "SEG_A") & (pred == "SEG_C")).sum())
    return {
        "thr_a": thr_a, "thr_c": thr_c,
        "cost": cost, "cost_per_hcp": cost / len(y_true),
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "recall_C": n_C_correct / max(n_C, 1),
        "c_lost_as_a_pct": n_C_lost_as_A / max(n_C, 1),
        "n_C_lost_as_a": n_C_lost_as_A,
        "n_A_to_c": n_A_to_C,
        "pred": pred,
    }


def grid_search(y_true, P_A, P_C, cost_matrix,
                thr_a_grid, thr_c_grid, verbose=False):
    rows = []
    best = None
    for ta in thr_a_grid:
        for tc in thr_c_grid:
            r = evaluate(y_true, P_A, P_C, ta, tc, cost_matrix)
            rows.append({k: v for k, v in r.items() if k != "pred"})
            if best is None or r["cost"] < best["cost"]:
                best = r
        if verbose:
            print(f"    thr_A = {ta:.3f} done", flush=True)
    return best, pd.DataFrame(rows)


# ============================ Plotting ============================
def plot_grid(df_grid, thr_a_grid, thr_c_grid, heur, opt, cost_label, output_path):
    pivot = df_grid.pivot(index="thr_a", columns="thr_c", values="cost")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1) Cost heatmap
    ax = axes[0, 0]
    im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                   cmap="viridis_r",
                   extent=[thr_c_grid.min(), thr_c_grid.max(),
                           thr_a_grid.min(), thr_a_grid.max()])
    ax.set_xlabel("Threshold for SEG_C  (P_C ≥ thr_c)")
    ax.set_ylabel("Threshold for SEG_A  (P_A ≥ thr_a)")
    ax.set_title(f"Total business cost  ({cost_label})\nLower = better")
    plt.colorbar(im, ax=ax, label="cost")
    ax.scatter([HEURISTIC_THR_C], [HEURISTIC_THR_A], marker="o",
               s=140, c="red", edgecolor="white", linewidth=2,
               label=f"Heuristic (cost {heur['cost']:.0f})", zorder=5)
    ax.scatter([opt["thr_c"]], [opt["thr_a"]], marker="*",
               s=320, c="lime", edgecolor="black", linewidth=1.5,
               label=f"Optimum (cost {opt['cost']:.0f})", zorder=5)
    ax.legend(loc="upper right")

    # 2) Recall_C heatmap
    ax = axes[0, 1]
    pivot_rc = df_grid.pivot(index="thr_a", columns="thr_c", values="recall_C")
    im = ax.imshow(pivot_rc.values, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[thr_c_grid.min(), thr_c_grid.max(),
                           thr_a_grid.min(), thr_a_grid.max()])
    ax.set_xlabel("thr_C")
    ax.set_ylabel("thr_A")
    ax.set_title("Recall SEG_C\nHigher = fewer prescribers missed")
    plt.colorbar(im, ax=ax, label="recall_C")
    ax.scatter([HEURISTIC_THR_C], [HEURISTIC_THR_A], marker="o", s=140, c="red",
               edgecolor="white", linewidth=2, zorder=5)
    ax.scatter([opt["thr_c"]], [opt["thr_a"]], marker="*", s=320, c="black",
               edgecolor="white", linewidth=1.5, zorder=5)

    # 3) Accuracy heatmap
    ax = axes[1, 0]
    pivot_acc = df_grid.pivot(index="thr_a", columns="thr_c", values="accuracy")
    im = ax.imshow(pivot_acc.values, origin="lower", aspect="auto", cmap="Blues",
                   extent=[thr_c_grid.min(), thr_c_grid.max(),
                           thr_a_grid.min(), thr_a_grid.max()])
    ax.set_xlabel("thr_C")
    ax.set_ylabel("thr_A")
    ax.set_title("Plain accuracy\n(NOT the optimization target)")
    plt.colorbar(im, ax=ax, label="accuracy")
    ax.scatter([HEURISTIC_THR_C], [HEURISTIC_THR_A], marker="o", s=140, c="red",
               edgecolor="white", linewidth=2, zorder=5)
    ax.scatter([opt["thr_c"]], [opt["thr_a"]], marker="*", s=320, c="black",
               edgecolor="white", linewidth=1.5, zorder=5)

    # 4) C-lost heatmap
    ax = axes[1, 1]
    pivot_lost = df_grid.pivot(index="thr_a", columns="thr_c",
                               values="c_lost_as_a_pct")
    im = ax.imshow(pivot_lost.values, origin="lower", aspect="auto",
                   cmap="Reds",
                   extent=[thr_c_grid.min(), thr_c_grid.max(),
                           thr_a_grid.min(), thr_a_grid.max()])
    ax.set_xlabel("thr_C")
    ax.set_ylabel("thr_A")
    ax.set_title("SEG_C → A losses (catastrophic errors)\nLower = better")
    plt.colorbar(im, ax=ax, label="C→A loss rate")
    ax.scatter([HEURISTIC_THR_C], [HEURISTIC_THR_A], marker="o", s=140, c="black",
               edgecolor="white", linewidth=2, zorder=5)
    ax.scatter([opt["thr_c"]], [opt["thr_a"]], marker="*", s=320, c="lime",
               edgecolor="black", linewidth=1.5, zorder=5)

    fig.suptitle("Threshold optimization grid · ordinal HCP model",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


# ============================ Reporting ============================
def banner(text, char="=", width=78):
    print(char * width)
    print(text)
    print(char * width)


def report_comparison(heur, opt, n_total):
    banner("RESULTS — Heuristic vs Optimized Thresholds")
    print(f"\n  HEURISTIC (current model):  thr_A = {HEURISTIC_THR_A:.2f}  thr_C = {HEURISTIC_THR_C:.2f}")
    print(f"  OPTIMIZED (cost-minimum):   thr_A = {opt['thr_a']:.3f}  thr_C = {opt['thr_c']:.3f}")

    rows = [
        ("Total business cost",      f"{heur['cost']:>10.1f}", f"{opt['cost']:>10.1f}"),
        ("Cost per HCP",             f"{heur['cost_per_hcp']:>10.4f}", f"{opt['cost_per_hcp']:>10.4f}"),
        ("Accuracy",                 f"{heur['accuracy']*100:>9.2f}%", f"{opt['accuracy']*100:>9.2f}%"),
        ("Balanced accuracy",        f"{heur['balanced_accuracy']*100:>9.2f}%", f"{opt['balanced_accuracy']*100:>9.2f}%"),
        ("Recall SEG_C",             f"{heur['recall_C']*100:>9.2f}%", f"{opt['recall_C']*100:>9.2f}%"),
        ("SEG_C → A losses (rate)",  f"{heur['c_lost_as_a_pct']*100:>9.2f}%", f"{opt['c_lost_as_a_pct']*100:>9.2f}%"),
        ("SEG_C → A losses (count)", f"{heur['n_C_lost_as_a']:>10d}", f"{opt['n_C_lost_as_a']:>10d}"),
        ("SEG_A → C errors (count)", f"{heur['n_A_to_c']:>10d}", f"{opt['n_A_to_c']:>10d}"),
    ]
    print()
    print(f"  {'Metric':<28s} {'Heuristic':>12s} {'Optimized':>12s}   {'Δ':<14s}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}   {'-'*14}")
    for name, h, o in rows:
        if "%" in h:
            h_val = float(h.strip().rstrip('%'))
            o_val = float(o.strip().rstrip('%'))
            delta = f"{o_val - h_val:+.2f} pp"
        elif "." in h:
            h_val = float(h.strip()); o_val = float(o.strip())
            delta = f"{o_val - h_val:+.4f}"
        else:
            h_val = int(h.strip()); o_val = int(o.strip())
            delta = f"{o_val - h_val:+d}"
        print(f"  {name:<28s} {h:>12s} {o:>12s}   {delta:<14s}")

    print()
    cost_savings_pct = (heur["cost"] - opt["cost"]) / heur["cost"] * 100
    c_saved = heur["n_C_lost_as_a"] - opt["n_C_lost_as_a"]
    extra_a_to_c = opt["n_A_to_c"] - heur["n_A_to_c"]
    print(f"  Cost reduction: {cost_savings_pct:+.2f}% ")
    if c_saved > 0:
        print(f"  → {c_saved} additional SEG_C prescribers retained (C→A losses prevented)")
    elif c_saved < 0:
        print(f"  → {-c_saved} more SEG_C prescribers lost (worse on C-recovery)")
    if extra_a_to_c > 0:
        print(f"  → {extra_a_to_c} additional SEG_A doctors mis-targeted as SEG_C (extra cost paid)")
    elif extra_a_to_c < 0:
        print(f"  → {-extra_a_to_c} fewer SEG_A doctors wasted as SEG_C")


def confusion_table(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(VALID_LABELS))
    print(f"\n  {title}:")
    df = pd.DataFrame(cm,
        index=[f"true_{l}" for l in VALID_LABELS],
        columns=[f"pred_{l}" for l in VALID_LABELS])
    print(df.to_string())
    print(f"  Row totals (recall): " +
          " · ".join([f"{l}: {cm[i,i]/cm[i].sum()*100:.1f}%"
                      for i, l in enumerate(VALID_LABELS) if cm[i].sum() > 0]))


# ============================ Main ============================
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Optimize ordinal-model thresholds against a business cost.")
    p.add_argument("--input", required=True)
    p.add_argument("--label-col", default="ATSEG_first")
    p.add_argument("--id-col", default="NUEVO_ID")
    p.add_argument("--cost-c-to-a", type=float, default=5.0,
                   help="Cost of predicting A when true is C (catastrophic)")
    p.add_argument("--cost-a-to-c", type=float, default=1.0,
                   help="Cost of predicting C when true is A (costly)")
    p.add_argument("--cost-b-error", type=float, default=0.3,
                   help="Cost of any error involving B (acceptable)")
    p.add_argument("--thr-a-min", type=float, default=0.30)
    p.add_argument("--thr-a-max", type=float, default=0.95)
    p.add_argument("--thr-c-min", type=float, default=0.10)
    p.add_argument("--thr-c-max", type=float, default=0.70)
    p.add_argument("--thr-step", type=float, default=0.025)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--exclude-cols", nargs="*",
                   default=["WEEK_ID_first", "WEEK_ID_last", "WEEK_ID_count"])
    p.add_argument("--output-csv", default="threshold_search_results.csv")
    p.add_argument("--output-plot", default="threshold_grid.png")
    p.add_argument("--skip-sensitivity", action="store_true",
                   help="Skip the sensitivity analysis on cost ratios")
    args = p.parse_args()

    banner("THRESHOLD OPTIMIZATION FOR ORDINAL HCP MODEL")
    print(f"\nInput: {args.input}")
    print(f"Cost weights:   C→A = {args.cost_c_to_a}   A→C = {args.cost_a_to_c}   B-errors = {args.cost_b_error}")
    print(f"Grid: thr_A ∈ [{args.thr_a_min}, {args.thr_a_max}], thr_C ∈ [{args.thr_c_min}, {args.thr_c_max}], step = {args.thr_step}\n")

    # ---- Load data ----
    print("[1/4] Loading data...")
    df = pd.read_csv(args.input)
    print(f"  Shape: {df.shape}")
    df[args.label_col] = df[args.label_col].astype(str)
    is_labeled = df[args.label_col].isin(VALID_LABELS).values
    print(f"  Labeled HCPs: {is_labeled.sum():,}")
    if is_labeled.sum() == 0:
        sys.exit("  ERROR: no SEG_A/SEG_B/SEG_C labels found")

    df_lab = df[is_labeled].reset_index(drop=True)
    df_lab = add_ratios(df_lab)
    drop_cols = set([args.label_col, args.id_col] + list(args.exclude_cols))
    feat_cols = [c for c in df_lab.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df_lab[c])]
    print(f"  Features: {len(feat_cols)}")

    X = df_lab[feat_cols].fillna(0).values
    y = df_lab[args.label_col].values

    # ---- OOF probabilities ----
    print(f"\n[2/4] Computing OOF probabilities ({args.n_folds}-fold CV)...")
    P_A, P_B, P_C = oof_probabilities(X, y, n_splits=args.n_folds)
    print("  Done.")

    # ---- Build cost matrix ----
    cost_matrix = build_cost_matrix(args.cost_c_to_a, args.cost_a_to_c, args.cost_b_error)
    cost_label = f"C→A={args.cost_c_to_a}, A→C={args.cost_a_to_c}, B-err={args.cost_b_error}"

    # ---- Grid search ----
    print(f"\n[3/4] Grid search on thresholds...")
    thr_a_grid = np.round(np.arange(args.thr_a_min, args.thr_a_max + 1e-9, args.thr_step), 4)
    thr_c_grid = np.round(np.arange(args.thr_c_min, args.thr_c_max + 1e-9, args.thr_step), 4)
    print(f"  Combinations: {len(thr_a_grid)} × {len(thr_c_grid)} = {len(thr_a_grid)*len(thr_c_grid):,}")
    opt, df_grid = grid_search(y, P_A, P_C, cost_matrix, thr_a_grid, thr_c_grid)
    print(f"  Done.  Optimum: thr_A = {opt['thr_a']:.3f}, thr_C = {opt['thr_c']:.3f}, cost = {opt['cost']:.1f}")

    heur = evaluate(y, P_A, P_C, HEURISTIC_THR_A, HEURISTIC_THR_C, cost_matrix)

    # ---- Report ----
    print()
    report_comparison(heur, opt, len(y))
    confusion_table(y, heur["pred"], "Confusion matrix — heuristic")
    confusion_table(y, opt["pred"],  "Confusion matrix — optimized")

    # ---- Save outputs ----
    df_grid.to_csv(args.output_csv, index=False)
    print(f"\n  Saved: {args.output_csv}  ({len(df_grid)} rows)")
    plot_grid(df_grid, thr_a_grid, thr_c_grid, heur, opt, cost_label, args.output_plot)
    print(f"  Saved: {args.output_plot}")

    # ---- Sensitivity analysis ----
    if not args.skip_sensitivity:
        print()
        banner("SENSITIVITY ANALYSIS — Optimum Across Cost Ratios", "-")
        print("\nIf the C→A:A→C cost ratio changes, does the optimal threshold pair move much?")
        print()
        sens_rows = []
        for ratio in [3, 5, 7, 10, 15]:
            cm_sens = build_cost_matrix(c_to_a=ratio, a_to_c=1.0, b_error=args.cost_b_error)
            opt_sens, _ = grid_search(y, P_A, P_C, cm_sens, thr_a_grid, thr_c_grid)
            sens_rows.append({
                "ratio_C_to_A:A_to_C": f"{ratio}:1",
                "best_thr_A": round(opt_sens["thr_a"], 3),
                "best_thr_C": round(opt_sens["thr_c"], 3),
                "recall_C":   f"{opt_sens['recall_C']*100:.1f}%",
                "C->A_lost":  opt_sens["n_C_lost_as_a"],
                "A->C_count": opt_sens["n_A_to_c"],
                "accuracy":   f"{opt_sens['accuracy']*100:.1f}%",
            })
        sens_df = pd.DataFrame(sens_rows)
        print(sens_df.to_string(index=False))
        sens_df.to_csv("threshold_sensitivity.csv", index=False)
        print(f"\n  Saved: threshold_sensitivity.csv")
        print(f"\n  Interpretation: if the optimum moves a lot when the cost ratio changes,")
        print(f"  it means the choice depends heavily on how you weight the errors.")
        print(f"  If it stays roughly the same, the optimum is ROBUST.")

    print("\nDone.")


if __name__ == "__main__":
    main()