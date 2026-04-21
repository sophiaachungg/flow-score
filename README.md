# Flow Limitation Classification

Breath-by-breath binary classification of inspiratory flow limitation (FL) from polysomnographic airflow signals, with SHAP-based explainability and a human-in-the-loop (HITL) clinical web interface.

> **Status:** Research prototype — not validated for clinical use. External test set validation is required before deployment.

---

## Background

The apnea-hypopnea index (AHI) is the dominant metric for OSA severity but fails to capture sub-apneic pharyngeal airflow obstruction or flow limitation (FL) that occurs between scored events. FL frequency is largely independent of AHI, correlates with elevated ventilatory drive and daytime symptoms, and is a prerequisite for phenotyping patients who may respond to non-CPAP therapies (mandibular advancement, hypoglossal nerve stimulation, positional therapy).

This project implements an automated ML pipeline for FL detection using 17 amplitude-independent, phase-separated airflow shape features grounded in the physiological taxonomy of Mann et al. (see [References](#references)).

**Key result:** Our LightGBM achieves ROC-AUC = 0.887 and Cohen's κ = 0.524 under leave-one-patient-out cross-validation (LOPO-CV) on 6,993 labeled breaths from 84 patients. Our EfficientNet-B0 model trained on waveform images generated from channel signals achieves ROC-AUC 0.922 and Cohen's κ = 0.618. These results are competitive with the Mann et al. 2021 benchmark despite a substantially smaller cohort and a more challenging binary label scheme. 

---

## Repository Structure

```
fl-classification/
│
├── notebooks/
│   ├── 01-compare-to-baseline/          # Primary benchmark pipeline
│   │   ├── 01_data_preparation.ipynb     # Signal ingestion, breath segmentation, feature engineering
│   │   ├── 02_model_training.ipynb       # LOPO-CV for LR, RF, XGBoost, LightGBM
│   │   └── 03_xai.ipynb                  # SHAP TreeExplainer: global, group-level, per-breath
│   │
│   ├── 02-remove-correlated-features/   # Ablation: effect of feature redundancy pruning
│   │   ├── 01_data_preparation.ipynb     # Feature correlation filtering (Spearman ρ > 0.85)
│   │   ├── 02_model_training.ipynb       # Re-benchmark on pruned feature set
│   │   └── 03_xai_clinical.ipynb         # XAI on pruned model for clinical interpretability
│   │
│   └── 03-image-classification/         # Image-based classification (EfficientNet-B0, ResNet18, SimpleCNN)
│       ├── 04_breath_image_dataset.ipynb # Render breaths as images for CNN input
│       ├── 05_image_classification.ipynb # CNN training and LOPO-CV evaluation
│       └── 06_xai_image_models.ipynb     # GradCAM / LIME explanations for image models
│
├── fl_app/                        # Human-in-the-loop clinical interface
│   ├── frontend/
│   │   └── index.html                 # Single-page app; patient browser, breath viewer, SHAP panel
│   │
│   └── backend/
│       ├── main.py                    # FastAPI app: /patients, /breaths, /classify, /explain endpoints
│       ├── model_logreg.pkl           # Serialized logistic regression model (11-feature pruned set)
│       ├── scaler_logreg.pkl          # Corresponding StandardScaler
│       └── data/
│           ├── processed/                # Per-patient processed data (see Data Format below)
│           └── saved_sessions/           # Clinician session exports (JSON)
│
├── models/                           # Serialized trained models from compare-to-baseline
│   ├── lightgbm_lopo.pkl                 # LightGBM (best performer, AUC = 0.887, κ = 0.524)
│   ├── xgboost_lopo.pkl
│   ├── random_forest_lopo.pkl
│   ├── logreg_lopo.pkl
│   └── shap_explainer_lightgbm.pkl       # Pre-computed SHAP TreeExplainer object
│
└── README.md
```

---

## Data Format

Each patient's processed data lives in its own subdirectory under `fl_app/backend/data/processed/` (and equivalently for notebook pipelines). The naming convention is `{PatientID}_{YYYY-MM}/`.

```
data/processed/
└── patient_id/
    ├── patient_id_signals_100Hz.parquet   # Continuous airflow (and optionally effort) signal at 100 Hz
    ├── patient_id_signals_meta.json        # Recording metadata (sample rate, channel names, duration)
    ├── patient_id_breaths.xlsx             # Per-breath feature table with labels (annotator ground truth)
    └── patient_id_breaths.csv             # Same as above, CSV format for notebook compatibility
```

### `*_signals_100Hz.parquet` columns

| Column | Description |
|--------|-------------|
| `time_s` | Time in seconds from recording start |
| `flow` | Airflow signal (pneumotachographic or nasal pressure, amplitude not normalized) |
| `effort` | Respiratory effort channel (optional; used for breath segmentation) |

### `*_signals_meta.json` fields

```json
{
  "patient_id": "SC_2026-04",
  "sample_rate_hz": 100,
  "channels": ["flow", "effort"],
  "duration_s": 28413
}
```

### `*_breaths.csv` / `*_breaths.xlsx` columns

| Column | Type | Description |
|--------|------|-------------|
| `breath_id` | int | Monotonic breath index within the recording |
| `start_s`, `end_s` | float | Breath boundaries in seconds |
| `insp_start_s`, `insp_end_s` | float | Inspiratory phase boundaries |
| `exp_start_s`, `exp_end_s` | float | Expiratory phase boundaries |
| `label` | str | Ground-truth class: `FL`, `NFL`, `Apnea`, `Remove`, `PFL` |
| `label_binary` | int | 1 = FL, 0 = NFL (apnea/artifact/PFL excluded) |
| `quad_insp`, `quad_insp_50`, ... | float | 17 amplitude-independent shape features (see Feature Engineering) |
| `fl_prob` | float | Model probability score (populated after classification) |
| `fl_pred` | int | Binary prediction at recommended threshold 0.35 |

---

## Feature Engineering

17 features across 6 physiological categories, following the taxonomy of Mann et al. (2019, 2021). All features are amplitude-independent, computed on phase-normalized waveforms.

| Category | Features | Localizable? |
|----------|----------|-------------|
| Scooping | `quad_insp`, `quad_insp_50`, `quad_exp`, `area_under_peaks_insp` | ✅ Waveform |
| Spectral flutter | `power_5to12_insp`, `power_5to12_exp` | ❌ Spectral panel |
| Flatness | `flatness_insp_75`, `flatness_insp_90` | ✅ Waveform |
| Asymmetry | `insp_peak_position`, `pif_pef_ratio` | ✅ Waveform |
| Phase timing | `ie_ratio`, `insp_duty`, `breath_duration` | ✅ Waveform |
| Shape variability | `exp_cv`, `insp_skew`, `exp_skew`, `insp_kurt` | ❌ Spectral panel |

**Key finding:** Spectral flutter (`power_5to12_exp`, `power_5to12_insp`) ranks first and second globally by mean |SHAP value|, accounting for 71% of breaths as the dominant SHAP group. Removing the flutter group drops ROC-AUC by ΔAUC = 0.015 — more than twice the next largest contribution.

---

## Model Performance (LOPO-CV, n = 84 patients)

| Model | AUC | Cohen's κ |
|-------|-----|-----------|
| Logistic Regression | 0.883 | 0.507 |
| Random Forest | 0.884 | 0.518 |
| XGBoost | 0.885 | 0.521 |
| **LightGBM** | **0.887** | **0.524** |
| Mann et al. 2021† | — | 0.529 |

† Mann et al. used a 3-class ordinal scheme, 23 features, 40 patients, and 117,871 breaths. Comparison is approximate.

**Recommended deployment threshold:** 0.35 (sensitivity = 0.841, specificity = 0.673). The 0.50 default threshold yields sensitivity = 0.750. The lower threshold reflects the clinical asymmetry: a false negative leaves a patient with significant pharyngeal obstruction undetected; a false positive triggers technician review.
 
### Image-based classifiers (60/12/12 patient train/val/test split, n = 7,007 images)
 
| Model | Accuracy | F1 | AUC | Cohen's κ |
|-------|----------|----|-----|-----------|
| **EfficientNet-B0** | **0.815** | **0.849** | **0.922** | **0.618** |
| ResNet18 | 0.795 | 0.836 | 0.908 | 0.575 |
| SimpleCNN | 0.748 | 0.802 | 0.875 | 0.474 |
 
EfficientNet-B0 outperforms all feature-based classifiers on every metric. Note the evaluation protocols differ (patient-split holdout vs. LOPO-CV), so direct numeric comparison should be made cautiously. The image approach sacrifices the feature-level interpretability that motivated the SHAP analysis in the feature-based pipeline. However, implementing Grad-CAM for saliency maps could be potentially useful for clinician interpretability.
 
---
 
## Running the HITL App
 
The web app requires Python 3.12+ and the packages listed below.
 
```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy scipy scikit-learn lightgbm shap openpyxl pyarrow
 
# Start the backend (from fl_app_v2/backend/)
cd fl_app_v2/backend
uvicorn main_v2:app --reload --port 8000
 
# Open the frontend
open fl_app_v2/frontend/index_v3.html
# or serve it via any static file server
```
 
The backend exposes:
- `GET /patients` — list available patient directories
- `GET /breaths?patient_id=...` — return breath table for a patient
- `POST /classify` — run FL classification on a breath payload
- `POST /explain` — return per-feature SHAP contributions for a breath
---

---

## References

1. **Mann DL, Georgeson T, Landry SA, et al.** "Frequency of flow limitation using airflow shape." *SLEEP*, 44(12): zsab170, 2021. https://doi.org/10.1093/sleep/zsab170

2. **Mann DL, Azarbarzin A, Vena D, et al.** "Airflow shape is associated with the degree of pharyngeal obstruction during sleep." *Eur Respir J*, 54(1): 1802262, 2019. https://doi.org/10.1183/13993003.02262-2018

3. **Genta PR, Sands SA, Butler JP, et al.** "Airflow shape is associated with the pharyngeal structure causing OSA." *CHEST*, 152(3): 537–546, 2017. https://doi.org/10.1016/j.chest.2017.06.017

4. **Das N, Happaerts S, Gyselinck I, et al.** "Collaboration between explainable artificial intelligence and pulmonologists improves the accuracy of pulmonary function test interpretation." *Eur Respir J*, 61(5): 2201720, 2023. https://doi.org/10.1183/13993003.01720-2022

---

## Authors

Sophia Chung & Ash Ren — Vanderbilt University, Data Science Institute

Yike Li, MD, PhD - Vanderbilt University Medical Center, Department of Otolaryngology
