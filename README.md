# Pfizer HCP Segmentation — Velsipity

Classifying healthcare providers (HCPs) into three actionable segments by their
propensity to prescribe **Velsipity** (oral therapy for ulcerative colitis), so
that the ~15K untyped HCPs can be targeted as effectively as the small set that
were originally interviewed.

> Capstone project · Tec de Monterrey × Pfizer Global Commercial Analytics

---

## The segments

| Segment | Meaning | Share (labeled) |
|---------|---------|-----------------|
| **SEG_A** | Low propensity — not a target | ~54% |
| **SEG_B** | Medium / ambiguous | ~28% |
| **SEG_C** | High propensity — prescriber | ~18% |

## The chosen model

A **business-calibrated two-stage ordinal XGBoost**. Two cascading binary
classifiers estimate `P(>=B)` and `P(>=C)`; class probabilities are recovered
and turned into a decision with asymmetric thresholds plus a dominance rule:

```
P(A) >= 0.70   -> SEG_A
P(C) >= 0.30   -> SEG_C
P(C) >  P(B)   -> SEG_C        (dominance rule)
otherwise      -> SEG_B
```

This deliberately trades a few points of overall accuracy for **higher recall on
SEG_C and far fewer catastrophic C->A errors** (a real prescriber wrongly
dropped). An *argmax baseline* is kept side-by-side for comparison. Performance
is reported from honest 5-fold out-of-fold predictions.

Why not a fancier model? The accuracy ceiling (~58–59% balanced) is **data-driven**,
not algorithmic: SEG_C and SEG_A overlap almost completely in feature space and
the labels carry noise. The lever that adds business value is **threshold
calibration**, not more model complexity.

---

## Repository layout

```
pfizer-hcp-segmentation/
├── README.md
├── requirements.txt
├── src/                     ← shared core (single source of truth)
│   ├── config.py            ← constants: thresholds, params, columns, palette
│   ├── features.py          ← load_data, add_features, feature selection
│   ├── model.py             ← ordinal + argmax models, OOF cross-val, SHAP
│   ├── metrics.py           ← balanced / weighted / cost-sensitive metrics
│   └── intelligence.py      ← centroids, prototypes, levers, churn, insights
├── pipeline/
│   ├── aggregate_doctors.py ← raw weekly CSV → data/processed/doctors_aggregated.csv
│   └── precompute_data.py   ← doctors_aggregated.csv → precomputed_results.pkl
├── app/
│   └── pfizer_dashboard.py  ← Streamlit dashboard (STATIC: loads the pkl) — the presented build
├── reports/
│   ├── segment_intelligence.py   ← static HTML report: model + SHAP + CI +
│   │                               conversion + Segment Intelligence tab
│   └── counterfactual.py         ← "+1 visit" B→C simulation (plots)
├── assets/                  ← Pfizer logos (logo2.png, logo3.png)
├── data/
│   ├── raw/                 ← Data_2026.csv (weekly, not versioned)
│   └── processed/           ← doctors_aggregated.csv (one row per HCP)
├── precomputed_results.pkl  ← model results for the static dashboard (not versioned)
└── outputs/                 ← generated HTML / CSV (not versioned)
```

Everything imports the model, features and metrics from `src/`, so the
dashboard, the reports and the counterfactual analysis are guaranteed to use the
**exact same** model configuration.

---

## Setup

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Place the aggregated dataset at `data/processed/doctors_aggregated.csv`
(label column `ATSEG_first`, id column `NUEVO_ID`).

## Reproducing the whole pipeline from scratch

```bash
# 0) Put the raw weekly export at  data/raw/data_processed.csv
# 1) Aggregate 86 weekly rows per HCP into one row per HCP
python pipeline/aggregate_doctors.py
#    → data/processed/doctors_aggregated.csv  (~20,931 rows)

# 2) Train + cross-validate + SHAP + counterfactual + conversion, save the bundle
python pipeline/precompute_data.py
#    → precomputed_results.pkl  (loaded by the static dashboard)
```

## Running each deliverable

```bash
# A) Executive dashboard — the presented build (loads precomputed_results.pkl)
streamlit run app/pfizer_dashboard.py

# B) Static HTML report (segment intelligence + conversion strategy + SHAP/CI)
python reports/segment_intelligence.py --input data/processed/doctors_aggregated.csv

# C) Counterfactual: how many SEG_B doctors flip to SEG_C with more visits?
python reports/counterfactual.py --input data/processed/doctors_aggregated.csv
```

Random seed is fixed at **42** throughout for reproducibility.

---

## Notes

- All code, comments and generated outputs are in English.
- Data files and the precomputed pickle are git-ignored (size / confidentiality).
- `app/pfizer_dashboard.py` is the **static, presented build**: it loads
  `precomputed_results.pkl` (≈58 MB, kept at the repo root) and never trains.
- Data flow:
  `data/raw/data_processed.csv` → **aggregate_doctors** →
  `data/processed/doctors_aggregated.csv` → **precompute_data** →
  `precomputed_results.pkl` → **pfizer_dashboard.py**.
