"""
config.py
=========
Single source of truth for every constant used across the project:
model hyper-parameters, decision thresholds, column names, the key feature
list shown in the dashboards, and the Pfizer colour palette.

Importing from here (instead of redefining constants inside each script)
guarantees that the Streamlit app, the HTML reports and the counterfactual
analysis all use *exactly* the same model configuration.

Tec de Monterrey x Pfizer Global Commercial Analytics
"""

# ────────────────────────────────────────────────────────────────────────
# Reproducibility & labels
# ────────────────────────────────────────────────────────────────────────
SEED = 42

#: The three actionable segments, in ordinal order (low -> high propensity).
VALID_LABELS = ("SEG_A", "SEG_B", "SEG_C")

#: Column holding the segment label in the aggregated dataset.
LABEL_COL = "ATSEG_first"

#: Column holding the unique HCP identifier.
ID_COL = "NUEVO_ID"


# ────────────────────────────────────────────────────────────────────────
# Decision thresholds (the business-calibrated, chosen model)
# ────────────────────────────────────────────────────────────────────────
#: P(A) must clear this high bar before we "abandon" a doctor as a non-target.
THR_A = 0.70

#: P(C) only needs to clear this low bar to flag a doctor as a prescriber.
THR_C = 0.30

#: Extra weight given to SEG_C samples when training the P(>=C) stage, so the
#: model pays more attention to the rare, high-value prescriber class.
SEG_C_WEIGHT = 2.0


# ────────────────────────────────────────────────────────────────────────
# Model hyper-parameters
# ────────────────────────────────────────────────────────────────────────
#: Two-stage ordinal XGBoost (used for both P(>=B) and P(>=C) classifiers).
ORDINAL_PARAMS = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)

#: Single multiclass XGBoost used only as the "argmax baseline" for comparison.
ARGMAX_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=3,
    random_state=SEED,
    n_jobs=-1,
    eval_metric="mlogloss",
)


# ────────────────────────────────────────────────────────────────────────
# Feature handling
# ────────────────────────────────────────────────────────────────────────
#: Substrings of columns that must never be used as model inputs (leakage,
#: identifiers, or alternative segment encodings). Used by select_features().
DATA_EXCLUDE = [
    LABEL_COL, ID_COL,
    "ATSEG", "SEGMENT", "PROB_SEG", "CONFIDENCE", "CLUSTER",
    "WEEK_ID_first", "WEEK_ID_last", "WEEK_ID_count",
]

#: Key features surfaced in the dashboards' Doctor Explorer.
#: Each tuple is (column, friendly_name, description, is_actionable):
#:   is_actionable=True  -> engagement levers Pfizer can directly influence
#:   is_actionable=False -> the doctor's own prescribing signal (an outcome)
KEY_FEATURES = [
    ("DETAILS_sum",   "Sales detail visits",        "Pfizer rep visits to this HCP",            True),
    ("SAMPLES_sum",   "Drug samples",               "Samples delivered to HCP",                  True),
    ("UC_TRX_sum",    "UC prescriptions (TRx)",     "Total Rx for ulcerative colitis",           False),
    ("UC_NRX_sum",    "UC new prescriptions (NRx)", "Patients newly starting UC therapy",        False),
    ("ORAL_TRX_sum",  "Oral therapy Rx",            "Oral UC treatments - Velsipity's category", False),
    ("IL23_TRX_sum",  "IL-23 inhibitor Rx",         "Competitor biologics for UC",               False),
    ("TOTAL_TRX_sum", "Total Rx volume",            "Overall prescribing activity",              False),
]

#: How many top SHAP features to expose per doctor.
N_SHAP_TOP = 8

#: Top predicted-B doctors (ranked by P_C) considered for the conversion tab.
N_CONVERSION_CANDIDATES = 300


# ────────────────────────────────────────────────────────────────────────
# Branding (Pfizer)
# ────────────────────────────────────────────────────────────────────────
PFIZER_BLUE       = "#0070BF"
PFIZER_DARK_BLUE  = "#003B71"
PFIZER_LIGHT_BLUE = "#00AFF0"
PFIZER_ORANGE     = "#F47B20"
PFIZER_GREEN      = "#00A651"
PFIZER_PURPLE     = "#7C3F98"
PFIZER_GRAY       = "#535554"

#: Consistent colour per segment across every visualization.
SEG_COLORS = {
    "SEG_A": PFIZER_BLUE,
    "SEG_B": PFIZER_ORANGE,
    "SEG_C": PFIZER_PURPLE,
    "Unlabeled": "#CFD8DC",
    "Unlab.": "#CFD8DC",
    "": "#CFD8DC",
}
