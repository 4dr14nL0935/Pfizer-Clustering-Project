"""
src — shared core for the Pfizer HCP Segmentation project.

Every deliverable (Streamlit app, HTML reports, counterfactual analysis)
imports the model, features, metrics and constants from here, so there is
exactly one definition of each. Example:

    from src import (
        load_data, add_features, select_features,
        fit_ordinal, predict_ordinal, oof_with_ci,
        metrics_for, THR_A, THR_C,
    )
"""

from .config import (
    SEED, VALID_LABELS, LABEL_COL, ID_COL,
    THR_A, THR_C, SEG_C_WEIGHT,
    ORDINAL_PARAMS, ARGMAX_PARAMS,
    KEY_FEATURES, DATA_EXCLUDE,
    N_SHAP_TOP, N_CONVERSION_CANDIDATES,
    SEG_COLORS,
    PFIZER_BLUE, PFIZER_DARK_BLUE, PFIZER_LIGHT_BLUE,
    PFIZER_ORANGE, PFIZER_GREEN, PFIZER_PURPLE, PFIZER_GRAY,
)
from .features import (
    load_data, split_labeled, add_features,
    select_features, basic_feature_columns, compute_segment_stats,
)
from .model import (
    fit_ordinal, predict_ordinal,
    fit_argmax, predict_argmax,
    oof_with_ci, get_shap_values, top_shap_per_doctor,
)
from .metrics import (
    weighted_balanced_accuracy, cost_sensitive_accuracy,
    full_evaluation_report, metrics_for, DEFAULT_COST_MATRIX,
)
from .intelligence import (
    compute_segment_centroids, find_prototype_doctors,
    compute_business_levers, compute_churn_risks,
    generate_executive_insights,
)

__all__ = [
    # config
    "SEED", "VALID_LABELS", "LABEL_COL", "ID_COL",
    "THR_A", "THR_C", "SEG_C_WEIGHT",
    "ORDINAL_PARAMS", "ARGMAX_PARAMS",
    "KEY_FEATURES", "DATA_EXCLUDE",
    "N_SHAP_TOP", "N_CONVERSION_CANDIDATES", "SEG_COLORS",
    "PFIZER_BLUE", "PFIZER_DARK_BLUE", "PFIZER_LIGHT_BLUE",
    "PFIZER_ORANGE", "PFIZER_GREEN", "PFIZER_PURPLE", "PFIZER_GRAY",
    # features
    "load_data", "split_labeled", "add_features",
    "select_features", "basic_feature_columns", "compute_segment_stats",
    # model
    "fit_ordinal", "predict_ordinal", "fit_argmax", "predict_argmax",
    "oof_with_ci", "get_shap_values", "top_shap_per_doctor",
    # metrics
    "weighted_balanced_accuracy", "cost_sensitive_accuracy",
    "full_evaluation_report", "metrics_for", "DEFAULT_COST_MATRIX",
    # intelligence
    "compute_segment_centroids", "find_prototype_doctors",
    "compute_business_levers", "compute_churn_risks",
    "generate_executive_insights",
]
