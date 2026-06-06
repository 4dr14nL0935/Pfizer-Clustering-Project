"""
counterfactual.py
=================
Counterfactual analysis: if Pfizer increases sales visits (or samples) to the
doctors the model currently calls SEG_B, how many would be reclassified as
SEG_C (prescribers)?

METHODOLOGY:
  1. Train the ordinal model and get honest 5-fold OOF baseline predictions.
  2. Identify every doctor currently predicted SEG_B.
  3. For each scenario (+1 / +2 / +3 / +5 visits, or raise to the SEG_C
     median), modify the RAW feature, re-derive ALL engineered features, and
     re-predict. Counting flips B->C / B->A / stay-B.
  4. Repeat for SAMPLES_sum and for combined interventions; profile the
     "flippers" vs "stayers".

The model, feature engineering and thresholds all come from the shared `src/`
core, so this analysis is guaranteed consistent with the dashboard and report.

OUTPUT:
  - Confusion-matrix and movement plots in --output_dir
  - Console summary of conversions and first-visit ROI

USAGE:
  python reports/counterfactual.py --input data/processed/doctors_aggregated.csv

Tec de Monterrey x Pfizer Global Commercial Analytics
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Shared core — single source of truth for the model & features
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import (  # noqa: E402
    add_features, select_features,
    fit_ordinal, predict_ordinal,
    SEED, VALID_LABELS,
)

warnings.filterwarnings("ignore")


def simulate_intervention(
    df_raw,
    feat_cols,
    m1,
    m2,
    seg_b_idx,
    details_delta=None,
    details_target=None,
    samples_delta=None,
    samples_target=None,
    label="",
):
    """
    Modify raw features for SEG_B doctors → re-derive all features → re-predict.

    Key: we modify the RAW data BEFORE feature engineering, so all derived
    features (ratios like DETAILS/TRX, log_DETAILS, total_engagement, etc.)
    are correctly updated. This makes the simulation realistic.
    """
    df_sim = df_raw.copy()

    # Apply changes to raw features
    if details_delta is not None:
        df_sim.loc[df_sim.index[seg_b_idx], "DETAILS_sum"] += details_delta
    if details_target is not None:
        mask = df_sim.index[seg_b_idx]
        df_sim.loc[mask, "DETAILS_sum"] = df_sim.loc[mask, "DETAILS_sum"].clip(
            lower=details_target
        )
    if samples_delta is not None:
        df_sim.loc[df_sim.index[seg_b_idx], "SAMPLES_sum"] += samples_delta
    if samples_target is not None:
        mask = df_sim.index[seg_b_idx]
        df_sim.loc[mask, "SAMPLES_sum"] = df_sim.loc[mask, "SAMPLES_sum"].clip(
            lower=samples_target
        )

    # Re-derive ALL engineered features
    df_sim = add_features(df_sim)
    X_sim = df_sim[feat_cols].fillna(0).values

    # Re-predict ONLY the SEG_B doctors
    PA_s, PB_s, PC_s, pred_s = predict_ordinal(m1, m2, X_sim[seg_b_idx])

    b_to_c = int((pred_s == "SEG_C").sum())
    b_to_a = int((pred_s == "SEG_A").sum())
    b_stay = int((pred_s == "SEG_B").sum())

    return {
        "label": label,
        "b_to_c": b_to_c,
        "b_stay": b_stay,
        "b_to_a": b_to_a,
        "total": len(seg_b_idx),
        "pct_to_c": round(b_to_c / len(seg_b_idx) * 100, 1),
        "PC_mean": round(float(PC_s.mean()), 4),
        "pred": pred_s,
        "PC": PC_s,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    """Command-line entry point: run the full pipeline and write outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="doctors_aggregated.csv")
    parser.add_argument("--label_col", default="ATSEG_first")
    parser.add_argument("--id_col", default="NUEVO_ID")
    parser.add_argument("--output_dir", default="confusion_matrices")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  COUNTERFACTUAL SIMULATION — Velsipity HCP Segmentation")
    print("=" * 65)

    # ── LOAD DATA ──
    df_raw = pd.read_csv(args.input)
    print(f"\nData: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
    df_raw[args.label_col] = df_raw[args.label_col].astype(str)
    is_labeled = df_raw[args.label_col].isin(VALID_LABELS).values
    n_lab = int(is_labeled.sum())
    print(f"Labeled: {n_lab:,}  |  Unlabeled: {int((~is_labeled).sum()):,}")

    # Save raw values before engineering
    raw_details = df_raw["DETAILS_sum"].values.copy()
    raw_samples = df_raw["SAMPLES_sum"].values.copy()
    ids_all = df_raw[args.id_col].values
    y_all = df_raw[args.label_col].values

    # ── FEATURE ENGINEERING ──
    print("\nFeature engineering...")
    df = add_features(df_raw)
    exclude = [
        args.label_col,
        args.id_col,
        "ATSEG",
        "SEGMENT",
        "PROB_SEG",
        "CONFIDENCE",
        "CLUSTER",
        "WEEK_ID_first",
        "WEEK_ID_last",
        "WEEK_ID_count",
    ]
    feat_cols = select_features(df, exclude)
    print(f"Features selected: {len(feat_cols)}")

    X_lab = df.loc[is_labeled, feat_cols].fillna(0).values
    y_lab = y_all[is_labeled]

    # ── SEG_C MEDIANS ──
    c_mask = y_lab == "SEG_C"
    details_c_median = float(np.median(raw_details[is_labeled][c_mask]))
    samples_c_median = float(np.median(raw_samples[is_labeled][c_mask]))
    print(f"\nSEG_C medians:")
    print(f"  DETAILS_sum: {details_c_median:.1f}")
    print(f"  SAMPLES_sum: {samples_c_median:.1f}")

    # ── TRAIN MODEL ──
    print("\nTraining ordinal model on all labeled data...")
    m1, m2 = fit_ordinal(X_lab, y_lab)

    # ── OOF BASELINE ──
    print("Running 5-fold OOF for honest baseline...")
    X_all = df[feat_cols].fillna(0).values
    PA_base, PB_base, PC_base, pred_base = predict_ordinal(m1, m2, X_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred = np.empty(n_lab, dtype=object)
    oof_PC = np.zeros(n_lab)

    for fold, (tr, va) in enumerate(skf.split(X_lab, y_lab), 1):
        print(f"  Fold {fold}/5...", flush=True)
        om1, om2 = fit_ordinal(X_lab[tr], y_lab[tr])
        pa, pb, pc, pr = predict_ordinal(om1, om2, X_lab[va])
        oof_pred[va] = pr
        oof_PC[va] = pc

    # Merge OOF into full predictions
    pred_all = pred_base.copy()
    PC_all = PC_base.copy()
    lab_idx = np.where(is_labeled)[0]
    pred_all[lab_idx] = oof_pred
    PC_all[lab_idx] = oof_PC

    ba = balanced_accuracy_score(y_lab, oof_pred)
    rc = float((oof_pred[c_mask] == "SEG_C").mean())
    print(f"\nBaseline OOF: Bal.Acc={ba:.4f}, Recall_C={rc:.4f}")

    # ── IDENTIFY SEG_B ──
    seg_b_idx = np.where(pred_all == "SEG_B")[0]
    n_seg_b = len(seg_b_idx)
    print(f"\n{'=' * 65}")
    print(f"  SEG_B doctors to simulate: {n_seg_b:,}")
    print(f"{'=' * 65}")

    # ══════════════════════════════════════════════════════════════
    # RUN SCENARIOS
    # ══════════════════════════════════════════════════════════════

    all_results = []

    # --- DETAILS scenarios ---
    print(f"\n{'─' * 50}")
    print("SCENARIO GROUP: Increasing DETAILS_sum (sales visits)")
    print(f"{'─' * 50}")
    print(f"  {'Scenario':<40} {'→ C':>6} {'stay B':>8} {'→ A':>6} {'% → C':>7}")
    print(f"  {'-' * 40} {'-----':>6} {'------':>8} {'-----':>6} {'------':>7}")

    for delta, label in [
        (1, "+1 visit"),
        (2, "+2 visits"),
        (3, "+3 visits"),
        (5, "+5 visits"),
    ]:
        r = simulate_intervention(
            df_raw, feat_cols, m1, m2, seg_b_idx, details_delta=delta, label=label
        )
        print(
            f"  {r['label']:<40} {r['b_to_c']:>6,} {r['b_stay']:>8,} {r['b_to_a']:>6,} {r['pct_to_c']:>6.1f}%"
        )
        all_results.append(r)

    r = simulate_intervention(
        df_raw,
        feat_cols,
        m1,
        m2,
        seg_b_idx,
        details_target=details_c_median,
        label=f"→ C median ({details_c_median:.0f} visits)",
    )
    print(
        f"  {r['label']:<40} {r['b_to_c']:>6,} {r['b_stay']:>8,} {r['b_to_a']:>6,} {r['pct_to_c']:>6.1f}%"
    )
    all_results.append(r)

    # --- SAMPLES scenarios ---
    print(f"\n{'─' * 50}")
    print("SCENARIO GROUP: Increasing SAMPLES_sum")
    print(f"{'─' * 50}")
    print(f"  {'Scenario':<40} {'→ C':>6} {'stay B':>8} {'→ A':>6} {'% → C':>7}")
    print(f"  {'-' * 40} {'-----':>6} {'------':>8} {'-----':>6} {'------':>7}")

    for delta, label in [(1, "+1 sample delivery"), (2, "+2 samples")]:
        r = simulate_intervention(
            df_raw, feat_cols, m1, m2, seg_b_idx, samples_delta=delta, label=label
        )
        print(
            f"  {r['label']:<40} {r['b_to_c']:>6,} {r['b_stay']:>8,} {r['b_to_a']:>6,} {r['pct_to_c']:>6.1f}%"
        )
        all_results.append(r)

    # --- COMBINED scenarios ---
    print(f"\n{'─' * 50}")
    print("SCENARIO GROUP: Combined interventions")
    print(f"{'─' * 50}")
    print(f"  {'Scenario':<40} {'→ C':>6} {'stay B':>8} {'→ A':>6} {'% → C':>7}")
    print(f"  {'-' * 40} {'-----':>6} {'------':>8} {'-----':>6} {'------':>7}")

    for dv, ds, label in [
        (1, 1, "+1 visit + 1 sample"),
        (2, 2, "+2 visits + 2 samples"),
    ]:
        r = simulate_intervention(
            df_raw,
            feat_cols,
            m1,
            m2,
            seg_b_idx,
            details_delta=dv,
            samples_delta=ds,
            label=label,
        )
        print(
            f"  {r['label']:<40} {r['b_to_c']:>6,} {r['b_stay']:>8,} {r['b_to_a']:>6,} {r['pct_to_c']:>6.1f}%"
        )
        all_results.append(r)

    r = simulate_intervention(
        df_raw,
        feat_cols,
        m1,
        m2,
        seg_b_idx,
        details_target=details_c_median,
        samples_target=samples_c_median,
        label="Both → C median",
    )
    print(
        f"  {r['label']:<40} {r['b_to_c']:>6,} {r['b_stay']:>8,} {r['b_to_a']:>6,} {r['pct_to_c']:>6.1f}%"
    )
    all_results.append(r)

    # ══════════════════════════════════════════════════════════════
    # FLIPPER ANALYSIS: Who are the +1 visit converts?
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print("  FLIPPER ANALYSIS: Who converts with +1 visit?")
    print(f"{'=' * 65}")

    # Re-run +1, +2, and median to get individual predictions
    # (stored as module-level vars so the confusion matrix section can reuse them)
    r1 = simulate_intervention(
        df_raw,
        feat_cols,
        m1,
        m2,
        seg_b_idx,
        details_delta=1,
        label="+1 visit (analysis)",
    )
    flippers_mask = r1["pred"] == "SEG_C"
    stayers_mask = r1["pred"] == "SEG_B"
    flippers_idx = seg_b_idx[flippers_mask]
    stayers_idx = seg_b_idx[stayers_mask]

    print(f"\n  Doctors who flip B→C with +1 visit: {len(flippers_idx):,}")
    print(f"    Avg P(C) before intervention:     {PC_all[flippers_idx].mean():.3f}")
    print(
        f"    Avg DETAILS_sum before:            {raw_details[flippers_idx].mean():.2f}"
    )
    print(
        f"    Avg UC_TRX_sum:                    {df_raw.loc[df_raw.index[flippers_idx], 'UC_TRX_sum'].mean():.2f}"
    )
    print(
        f"    % that were labeled (not untyped): {is_labeled[flippers_idx].mean() * 100:.1f}%"
    )

    print(f"\n  Doctors who stay B even with +1 visit: {len(stayers_idx):,}")
    print(f"    Avg P(C) before intervention:     {PC_all[stayers_idx].mean():.3f}")
    print(
        f"    Avg DETAILS_sum before:            {raw_details[stayers_idx].mean():.2f}"
    )

    # ── Incremental analysis ──
    r2 = simulate_intervention(
        df_raw,
        feat_cols,
        m1,
        m2,
        seg_b_idx,
        details_delta=2,
        label="+2 visits (analysis)",
    )
    flippers_2 = seg_b_idx[r2["pred"] == "SEG_C"]

    rM = simulate_intervention(
        df_raw,
        feat_cols,
        m1,
        m2,
        seg_b_idx,
        details_target=details_c_median,
        label="→ median (analysis)",
    )
    flippers_M = seg_b_idx[rM["pred"] == "SEG_C"]

    print(f"\n  Incremental conversions:")
    print(f"    +1 visit:   {len(flippers_idx):>5,} total")
    print(
        f"    +2 visits:  {len(flippers_2):>5,} total (+{len(flippers_2) - len(flippers_idx)} incremental)"
    )
    print(
        f"    → median:   {len(flippers_M):>5,} total (+{len(flippers_M) - len(flippers_idx)} vs +1)"
    )

    # ══════════════════════════════════════════════════════════════
    # P(C) DISTRIBUTION SHIFT
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print("  P(C) DISTRIBUTION SHIFT for SEG_B doctors")
    print(f"{'=' * 65}")

    pc_base = PC_all[seg_b_idx]
    pc_after = r1["PC"]
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0]
    bin_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"]

    print(
        f"\n  {'P(C) bucket':>10s}  {'Baseline':>10s}  {'After +1':>10s}  {'Shift':>10s}"
    )
    print(f"  {'-' * 10:>10s}  {'-' * 10:>10s}  {'-' * 10:>10s}  {'-' * 10:>10s}")
    for i in range(len(bins) - 1):
        n_base = int(((pc_base >= bins[i]) & (pc_base < bins[i + 1])).sum())
        n_after = int(((pc_after >= bins[i]) & (pc_after < bins[i + 1])).sum())
        diff = n_after - n_base
        sign = "+" if diff > 0 else ""
        print(
            f"  {bin_labels[i]:>10s}  {n_base:>10,}  {n_after:>10,}  {sign}{diff:>9,}"
        )

    # ══════════════════════════════════════════════════════════════
    # CONFUSION MATRICES — 4 DETAILS_sum SCENARIOS
    # ══════════════════════════════════════════════════════════════
    #
    # Only labeled doctors have ground truth, so all matrices are
    # computed on y_lab vs. the scenario predictions for that subset.
    #
    # For doctors NOT predicted as SEG_B in the baseline, their OOF
    # prediction is unchanged across scenarios.  For doctors that
    # WERE predicted as SEG_B, we substitute the scenario prediction.
    # ──────────────────────────────────────────────────────────────

    # Map full-dataset row index → position in the labeled subset
    labeled_full_indices = np.where(is_labeled)[0]  # shape (n_lab,)
    full_to_lab_pos = {int(fi): li for li, fi in enumerate(labeled_full_indices)}

    # Which positions in seg_b_idx are labeled, and their lab-subset positions
    seg_b_labeled_mask_full = np.array([is_labeled[i] for i in seg_b_idx], dtype=bool)
    seg_b_lab_subset_pos = np.array(
        [full_to_lab_pos[int(i)] for i in seg_b_idx[seg_b_labeled_mask_full]]
    )

    def build_labeled_preds(sim_result):
        """
        Return a prediction array of length n_lab.

        Start from OOF baseline; replace the SEG_B-labeled doctors'
        entries with the predictions returned by simulate_intervention.
        sim_result["pred"] is indexed over ALL seg_b_idx doctors —
        slice only the labeled ones via seg_b_labeled_mask_full.
        """
        preds = oof_pred.copy()
        preds[seg_b_lab_subset_pos] = sim_result["pred"][seg_b_labeled_mask_full]
        return preds

    def plot_cm(y_true, y_pred, scenario_title, filename):
        """Pretty-print and export a 3×3 confusion matrix."""
        labs = ["SEG_A", "SEG_B", "SEG_C"]
        cm = confusion_matrix(y_true, y_pred, labels=labs)
        ba = balanced_accuracy_score(y_true, y_pred)
        width = 12

        print(f"\n  ┌─ {scenario_title} ─")
        print(f"  │  Balanced Accuracy: {ba:.4f}")
        header = f"  │  {'':14s}" + "".join(f"{'Pred ' + l:>{width}}" for l in labs)
        print(header)
        sep = f"  │  {'':14s}" + ("─" * width) * 3
        print(sep)
        for i, row_label in enumerate(labs):
            row = f"  │  {'True ' + row_label:<14s}" + "".join(
                f"{cm[i, j]:>{width},}" for j in range(3)
            )
            print(row)
        print(f"  └{'─' * 60}")

        fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=160)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{scenario_title}\nExactitud balanceada: {ba:.4f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labs)))
        ax.set_yticks(range(len(labs)))
        ax.set_xticklabels([f"Pred {lab}" for lab in labs])
        ax.set_yticklabels([f"Real {lab}" for lab in labs])
        ax.set_xlabel("Etiqueta predicha")
        ax.set_ylabel("Etiqueta real")

        max_val = cm.max() if cm.size else 0
        thresh = max_val / 2 if max_val else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                )

        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════
    # SEGMENT MOVEMENT VISUALIZATIONS — 4 DETAILS_sum SCENARIOS
    # ══════════════════════════════════════════════════════════════
    # Shows, for ALL SEG_B doctors (labeled + unlabeled), how many
    # remain in SEG_B vs. are reclassified to SEG_C (or SEG_A).
    # ──────────────────────────────────────────────────────────────

    # Pfizer-aligned palette
    _C_STAY  = "#4A90D9"   # blue  → stays SEG_B
    _C_TO_C  = "#27AE60"   # green → flips to SEG_C (target)
    _C_TO_A  = "#BDC3C7"   # light gray → moves to SEG_A

    def plot_scenario_movement(sim_result, scenario_title, filename):
        """
        Donut + annotated breakdown for one scenario.
        Saves to output_dir/<filename>.
        """
        b_stay = sim_result["b_stay"]
        b_to_c = sim_result["b_to_c"]
        b_to_a = sim_result["b_to_a"]
        total  = sim_result["total"]

        sizes  = [b_stay, b_to_c, b_to_a]
        colors = [_C_STAY, _C_TO_C, _C_TO_A]
        labels = ["Permanece en SEG_B", "Mueve a SEG_C", "Mueve a SEG_A"]

        fig = plt.figure(figsize=(10, 5.5), dpi=160)
        fig.patch.set_facecolor("#F7F9FC")

        # ── Left: donut ──────────────────────────────────────────
        ax_donut = fig.add_axes([0.03, 0.10, 0.46, 0.80])
        wedges, _ = ax_donut.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
            counterclock=False,
        )
        # Centre text
        ax_donut.text(
            0, 0.12, f"{total:,}",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#2C3E50",
        )
        ax_donut.text(
            0, -0.18, "doctores SEG_B",
            ha="center", va="center", fontsize=9, color="#7F8C8D",
        )
        ax_donut.set_title(scenario_title, fontsize=13, fontweight="bold",
                           color="#2C3E50", pad=14)

        # ── Right: stat cards ────────────────────────────────────
        ax_stats = fig.add_axes([0.52, 0.08, 0.45, 0.84])
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis("off")

        card_data = [
            (b_stay, b_stay / total * 100, "Permanecen en SEG_B", _C_STAY),
            (b_to_c, b_to_c / total * 100, "Pasan a SEG_C", _C_TO_C),
            (b_to_a, b_to_a / total * 100, "Pasan a SEG_A", _C_TO_A),
        ]

        card_h = 0.24
        gap    = 0.06
        top    = 0.92

        for k, (count, pct, lab, col) in enumerate(card_data):
            y0 = top - k * (card_h + gap)
            # coloured left stripe
            ax_stats.add_patch(
                plt.Rectangle((0, y0 - card_h), 0.06, card_h,
                               transform=ax_stats.transData,
                               color=col, clip_on=False)
            )
            # card background
            ax_stats.add_patch(
                plt.Rectangle((0.07, y0 - card_h), 0.91, card_h,
                               transform=ax_stats.transData,
                               color="white", clip_on=False,
                               linewidth=0.8,
                               linestyle="-", edgecolor="#DDE3EA")
            )
            mid_y = y0 - card_h / 2
            ax_stats.text(0.12, mid_y + 0.045, lab,
                          fontsize=9.5, color="#555F6D", va="center")
            ax_stats.text(0.12, mid_y - 0.045,
                          f"{count:,} doctores",
                          fontsize=12, fontweight="bold", color="#2C3E50", va="center")
            ax_stats.text(0.93, mid_y,
                          f"{pct:.1f}%",
                          fontsize=15, fontweight="bold", color=col,
                          va="center", ha="right")

        fig.savefig(output_dir / filename, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Guardado: {output_dir / filename}")

    def plot_movement_summary(scenario_results, filename):
        """
        Horizontal stacked bar chart comparing all 4 scenarios.
        Each bar = total SEG_B doctors, split by outcome.
        """
        labels  = [r["label"] for r in scenario_results]
        stays   = np.array([r["b_stay"] for r in scenario_results])
        to_c    = np.array([r["b_to_c"] for r in scenario_results])
        to_a    = np.array([r["b_to_a"] for r in scenario_results])
        totals  = np.array([r["total"]  for r in scenario_results])

        n = len(scenario_results)
        fig, ax = plt.subplots(figsize=(12, 1.2 * n + 2.2), dpi=160)
        fig.patch.set_facecolor("#F7F9FC")
        ax.set_facecolor("#F7F9FC")

        y = np.arange(n)
        bar_h = 0.52

        bar_stay = ax.barh(y, stays,  height=bar_h, color=_C_STAY, label="Permanece en SEG_B")
        bar_c    = ax.barh(y, to_c,   height=bar_h, color=_C_TO_C, label="Pasa a SEG_C",
                           left=stays)
        bar_a    = ax.barh(y, to_a,   height=bar_h, color=_C_TO_A, label="Pasa a SEG_A",
                           left=stays + to_c)

        # Annotate each segment if wide enough
        for i in range(n):
            total = totals[i]
            # stay label
            if stays[i] / total > 0.06:
                ax.text(stays[i] / 2, i,
                        f"{stays[i]:,}\n({stays[i]/total*100:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            # to_c label
            if to_c[i] / total > 0.04:
                ax.text(stays[i] + to_c[i] / 2, i,
                        f"{to_c[i]:,}\n({to_c[i]/total*100:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            # to_a (usually small, skip if tiny)
            if to_a[i] / total > 0.04:
                ax.text(stays[i] + to_c[i] + to_a[i] / 2, i,
                        f"{to_a[i]:,}\n({to_a[i]/total*100:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="#555F6D", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10.5, color="#2C3E50")
        ax.set_xlabel("Número de doctores (SEG_B)", fontsize=10, color="#555F6D")
        ax.set_title("Movimiento de segmento por escenario — doctores SEG_B",
                     fontsize=13, fontweight="bold", color="#2C3E50", pad=14)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", colors="#95A5A6")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.label.set_color("#95A5A6")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.6, title="Resultado")

        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Guardado: {output_dir / filename}")

    # ── Build the 4 scenario results from already-computed vars ──
    # r1, r2, rM were computed in the FLIPPER ANALYSIS block above.
    # We also need the baseline (zero delta) for completeness.
    r0 = {
        "label": "Baseline (sin cambios)",
        "b_stay": n_seg_b,
        "b_to_c": 0,
        "b_to_a": 0,
        "total":  n_seg_b,
        "pct_to_c": 0.0,
    }

    four_scenarios = [
        (r0,  "Baseline — sin intervención",                      "mov_00_baseline.png"),
        (r1,  "+1 visita a doctores SEG_B",                       "mov_01_plus1_visit.png"),
        (r2,  "+2 visitas a doctores SEG_B",                      "mov_02_plus2_visits.png"),
        (rM,  f"Llevar a la mediana SEG_C ({details_c_median:.0f} visitas)", "mov_03_median.png"),
    ]

    print(f"\n{'=' * 65}")
    print("  VISUALIZACIONES — Movimiento de segmento por escenario")
    print(f"  Exportando a: {output_dir.resolve()}")
    print(f"{'=' * 65}")

    for sim_res, title, fname in four_scenarios:
        plot_scenario_movement(sim_res, title, fname)

    # Summary chart with the 4 main scenarios
    plot_movement_summary(
        [r0, r1, r2, rM],
        "mov_summary_4_scenarios.png",
    )

    print(f"\n{'=' * 65}")
    print("  MATRICES DE CONFUSIÓN — intervenciones sobre DETAILS_sum")
    print(f"  (solo doctores etiquetados, n={int(is_labeled.sum()):,})")
    print(f"{'=' * 65}")
    print(f"  Exportando imágenes a: {output_dir.resolve()}")

    # 1. Baseline — no intervention
    plot_cm(y_lab, oof_pred, "Escenario 0: línea base", "01_baseline.png")

    # 2. +1 visit
    preds_plus1 = build_labeled_preds(r1)
    plot_cm(
        y_lab,
        preds_plus1,
        "Escenario 1: +1 visita",
        "02_plus1_visit.png",
    )

    # 3. +2 visits
    preds_plus2 = build_labeled_preds(r2)
    plot_cm(
        y_lab,
        preds_plus2,
        "Escenario 2: +2 visitas",
        "03_plus2_visits.png",
    )

    # 4. → C median
    preds_median = build_labeled_preds(rM)
    plot_cm(
        y_lab,
        preds_median,
        f"Escenario 3: llevar a la mediana SEG_C ({details_c_median:.0f} visitas)",
        "04_seg_c_median.png",
    )

    # ══════════════════════════════════════════════════════════════
    # SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════════

    # Summary CSV
    summary_rows = []
    for r in all_results:
        summary_rows.append(
            {
                "scenario": r["label"],
                "B_to_C": r["b_to_c"],
                "B_stay_B": r["b_stay"],
                "B_to_A": r["b_to_a"],
                "total_B": r["total"],
                "pct_to_C": r["pct_to_c"],
            }
        )
    pd.DataFrame(summary_rows).to_csv("counterfactual_results.csv", index=False)
    print(f"\n✓ Saved: counterfactual_results.csv")

    # Flippers list
    flipper_df = pd.DataFrame(
        {
            "NUEVO_ID": ids_all[flippers_idx],
            "true_segment": y_all[flippers_idx],
            "was_labeled": is_labeled[flippers_idx],
            "P_C_before": np.round(PC_all[flippers_idx], 4),
            "P_C_after_plus1": np.round(r1["PC"][flippers_mask], 4),
            "DETAILS_sum_before": raw_details[flippers_idx],
            "UC_TRX_sum": df_raw.loc[df_raw.index[flippers_idx], "UC_TRX_sum"].values,
            "SAMPLES_sum": raw_samples[flippers_idx],
        }
    )
    flipper_df.to_csv("counterfactual_flippers.csv", index=False)
    print(f"✓ Saved: counterfactual_flippers.csv ({len(flipper_df):,} doctors)")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 65}")
    print("  EXECUTIVE SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total SEG_B doctors:           {n_seg_b:>6,}")
    print(f"  SEG_C median (DETAILS_sum):    {details_c_median:>6.1f}")
    print(f"")
    print(
        f"  +1 visit converts:             {len(flippers_idx):>6,}  ({len(flippers_idx) / n_seg_b * 100:.1f}%)"
    )
    print(
        f"  +2 visits converts:            {len(flippers_2):>6,}  ({len(flippers_2) / n_seg_b * 100:.1f}%)"
    )
    print(
        f"  → median converts:             {len(flippers_M):>6,}  ({len(flippers_M) / n_seg_b * 100:.1f}%)"
    )
    print(f"")
    print(f"  ★ First visit ROI:             {len(flippers_idx):>6,} doctors")
    print(
        f"  ★ Second visit incremental:    {len(flippers_2) - len(flippers_idx):>6,} doctors"
    )
    print(
        f"  ★ Diminishing returns ratio:   {(len(flippers_2) - len(flippers_idx)) / len(flippers_idx) * 100:.1f}%"
    )
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
