"""
metrics.py
==========
Evaluation metrics for the HCP segmentation model.

Beyond plain accuracy we care about *which* mistakes the model makes, because
they are not equally costly to the business:

  * C -> A  (a real prescriber labelled non-target)  = catastrophic, we stop
            engaging a doctor who would have prescribed Velsipity.
  * A -> C  (a non-prescriber labelled prescriber)    = recoverable, we only
            waste a sales visit.

The functions below quantify that asymmetry:
  1. balanced_accuracy        - simple mean of per-class recall (via sklearn).
  2. weighted_balanced_accuracy - per-class recall weighted toward SEG_C.
  3. cost_sensitive_accuracy  - 1.0 (perfect) to 0.0 (worst) using a cost matrix.
  4. metrics_for              - compact dict used by the dashboards.

Tec de Monterrey x Pfizer Global Commercial Analytics
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .config import VALID_LABELS


# ════════════════════════════════════════════════════════════════════════
# 1. Weighted balanced accuracy
# ════════════════════════════════════════════════════════════════════════
def weighted_balanced_accuracy(y_true, y_pred, class_weights=None):
    """Balanced accuracy with per-class importance weights.

    Standard balanced accuracy is the unweighted mean of per-class recall.
    This version lets SEG_C count more, reflecting that catching prescribers
    matters most.

    Parameters
    ----------
    y_true, y_pred : array-like
    class_weights : dict, optional
        e.g. ``{"SEG_A": 1, "SEG_B": 1, "SEG_C": 2}``. If None, weights are
        equal and the result equals standard balanced accuracy.

    Returns
    -------
    dict
        ``{"wba", "ba", "per_class"}`` where ``per_class[seg]`` holds recall,
        weight, weighted contribution and sample count.
    """
    classes = sorted(np.unique(y_true))
    if class_weights is None:
        class_weights = {c: 1.0 for c in classes}

    recalls = {}
    for cls in classes:
        mask = y_true == cls
        recalls[cls] = float((y_pred[mask] == cls).mean()) if mask.sum() else 0.0

    total_weight = sum(class_weights.get(c, 1.0) for c in classes)
    wba = sum(recalls[c] * class_weights.get(c, 1.0) for c in classes) / total_weight
    ba = sum(recalls[c] for c in classes) / len(classes)

    per_class = {
        cls: {
            "recall": recalls[cls],
            "weight": class_weights.get(cls, 1.0),
            "weighted_contribution": recalls[cls] * class_weights.get(cls, 1.0) / total_weight,
            "n_samples": int((y_true == cls).sum()),
        }
        for cls in classes
    }
    return {"wba": wba, "ba": ba, "per_class": per_class}


# ════════════════════════════════════════════════════════════════════════
# 2. Cost-sensitive accuracy
# ════════════════════════════════════════════════════════════════════════
#: Default business cost matrix: cost_matrix[true][pred].
#: 0 = correct; larger = worse. C->A is the most expensive error.
DEFAULT_COST_MATRIX = {
    #              pred_A  pred_B  pred_C
    "SEG_A": {"SEG_A": 0, "SEG_B": 1, "SEG_C": 2},   # A->C: wasted visits
    "SEG_B": {"SEG_A": 2, "SEG_B": 0, "SEG_C": 1},   # B->A: lost potential
    "SEG_C": {"SEG_A": 5, "SEG_B": 2, "SEG_C": 0},   # C->A: catastrophic
}


def cost_sensitive_accuracy(y_true, y_pred, cost_matrix=None):
    """Score in ``[0, 1]`` that penalises expensive errors more heavily.

    ``cost_score = 1 - total_cost / max_possible_cost`` where the max assumes
    every doctor of a class received that class's worst possible prediction.

    Returns
    -------
    dict
        cost_score, avg_cost_per_sample, total_cost, max_possible_cost,
        error_breakdown (per error type), and the confusion_matrix.
    """
    classes = list(VALID_LABELS)
    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n = len(y_true)

    total_cost = 0
    cost_breakdown = {}
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            c = cost_matrix[true_cls][pred_cls]
            count = int(cm[i, j])
            total_cost += c * count
            if count > 0 and c > 0:
                cost_breakdown[f"{true_cls}->{pred_cls}"] = {
                    "count": count,
                    "cost_per_error": c,
                    "total_cost": c * count,
                }

    max_cost = sum(
        max(cost_matrix[cls].values()) * int((y_true == cls).sum())
        for cls in classes
    )
    cost_score = 1 - (total_cost / max_cost) if max_cost > 0 else 1.0

    return {
        "cost_score": cost_score,
        "avg_cost_per_sample": total_cost / n if n else 0.0,
        "total_cost": total_cost,
        "max_possible_cost": max_cost,
        "error_breakdown": cost_breakdown,
        "confusion_matrix": cm,
    }


# ════════════════════════════════════════════════════════════════════════
# 3. Compact metric dict for the dashboards
# ════════════════════════════════════════════════════════════════════════
def metrics_for(y_true, pred, name=""):
    """Headline metrics for one set of predictions.

    Returns
    -------
    dict
        name, accuracy, balanced_accuracy, recall_C, c_lost_pct
        (= share of true SEG_C wrongly sent to SEG_A), confusion_matrix.
    """
    acc = float((pred == y_true).mean())
    ba = float(balanced_accuracy_score(y_true, pred))
    mc = y_true == "SEG_C"
    recall_c = float((pred[mc] == "SEG_C").mean()) if mc.sum() else 0.0
    c_lost = float(((y_true == "SEG_C") & (pred == "SEG_A")).sum() / max(mc.sum(), 1))
    cm = confusion_matrix(y_true, pred, labels=list(VALID_LABELS))
    return {
        "name": name,
        "accuracy": acc,
        "balanced_accuracy": ba,
        "recall_C": recall_c,
        "c_lost_pct": c_lost,
        "confusion_matrix": cm.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════
# 4. Full evaluation report (console)
# ════════════════════════════════════════════════════════════════════════
def full_evaluation_report(y_true, y_pred, seg_c_weight=2.0, print_report=True):
    """Print and return every metric in one place.

    Parameters
    ----------
    seg_c_weight : float
        Weight applied to SEG_C in the weighted balanced accuracy.
    print_report : bool
        If True, pretty-print the report to the console.

    Returns
    -------
    dict
        balanced_accuracy, weighted_balanced_accuracy, cost_sensitive_score,
        per_class, and the full cost_details.
    """
    weights = {"SEG_A": 1.0, "SEG_B": 1.0, "SEG_C": seg_c_weight}
    wba = weighted_balanced_accuracy(y_true, y_pred, weights)
    csa = cost_sensitive_accuracy(y_true, y_pred)

    if print_report:
        print("\n" + "=" * 60)
        print("  EVALUATION REPORT - HCP Segmentation Model")
        print("=" * 60)
        print(f"  Balanced Accuracy (standard):  {wba['ba']:.4f}")
        print(f"  Weighted Balanced Accuracy:    {wba['wba']:.4f}  (SEG_C x{seg_c_weight:.1f})")
        print(f"  Cost-Sensitive Score:          {csa['cost_score']:.4f}")
        print("\n  Per-class recall:")
        for cls in VALID_LABELS:
            info = wba["per_class"].get(cls)
            if info:
                print(f"    {cls}: recall={info['recall']:.4f}  n={info['n_samples']:,}")
        if csa["error_breakdown"]:
            print("\n  Costliest errors:")
            for name, e in sorted(csa["error_breakdown"].items(),
                                  key=lambda x: x[1]["total_cost"], reverse=True)[:5]:
                print(f"    {name}: {e['count']:,} x cost {e['cost_per_error']} = {e['total_cost']:,}")
        print(f"\n  Avg cost/sample: {csa['avg_cost_per_sample']:.3f}")
        print(f"  Total cost: {csa['total_cost']:,} / {csa['max_possible_cost']:,}")
        print("=" * 60)

    return {
        "balanced_accuracy": wba["ba"],
        "weighted_balanced_accuracy": wba["wba"],
        "cost_sensitive_score": csa["cost_score"],
        "per_class": wba["per_class"],
        "cost_details": csa,
    }
