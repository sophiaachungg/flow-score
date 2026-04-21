"""
Flow Limitation Classification — Backend API
================================================
FastAPI backend for the FL classification app.

Key changes from v1:
- Uses model_logreg.pkl + scaler_logreg.pkl (11-feature pruned set)
- Feature extraction updated to match 02-remove-correlated-features/01_data_preparation.ipynb
- /explain endpoint now returns exact LogReg per-feature contributions
  (coef_i × scaled_value_i) — no SHAP approximation needed for LogReg
- Contributions auto-computed on breath selection (no button required)
- FEATURE_DICT provides definitions + localisability metadata consumed by frontend
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(title="FL Classifier API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.80

MODEL_PATH  = Path("model_logreg.pkl")
SCALER_PATH = Path("scaler_logreg.pkl")

PROCESSED_DATA_DIR = Path("./data/processed")
SESSION_OUTPUT_DIR = Path("./data/saved_sessions")
SESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE DICTIONARY
# Single source of truth consumed by both backend and frontend.
# localisable: True  → frontend highlights a waveform region on hover
# localisable: False → frontend shows definition tooltip only
# highlight: describes what region/overlay to draw (used by frontend renderer)
# ─────────────────────────────────────────────

FEATURE_DICT = {
    "quad_insp_50": {
        "label": "Mid-inspiratory Scooping",
        "group": "Scooping",
        "localisable": True,
        "highlight": "insp_mid50",   # shade 25–75% of inspiratory phase
        "definition": (
            "Measures how much the middle 50% of the inspiratory flow waveform "
            "deviates from a matched parabola. A higher value means the flow "
            "dips below the expected parabolic shape in mid-inspiration — a "
            "hallmark of pharyngeal collapse (QuadI50, Mann et al.)."
        ),
    },
    "area_under_peaks_insp": {
        "label": "Inspiratory Scooping Depth",
        "group": "Scooping",
        "localisable": True,
        "highlight": "insp_peaks",   # draw peak-connecting line + shade area below
        "definition": (
            "Mean absolute deviation between the inspiratory flow signal and a "
            "line connecting its local peaks, normalised by peak inspiratory flow. "
            "Captures irregular mid-inspiratory dips not captured by the global "
            "parabolic fit (AreaUnderPeaksI, Mann et al.)."
        ),
    },
    "power_5to12_insp": {
        "label": "Inspiratory Flutter (5–12 Hz)",
        "group": "Flutter",
        "localisable": False,
        "highlight": None,
        "definition": (
            "Fraction of total spectral power in the 5–12 Hz frequency band "
            "during inspiration. Flutter in this range reflects rapid, "
            "oscillatory airflow caused by partial pharyngeal collapse. "
            "Amplitude-independent (computed as a ratio). Non-localisable: "
            "applies to the whole inspiratory phase."
        ),
    },
    "power_5to12_exp": {
        "label": "Expiratory Flutter (5–12 Hz)",
        "group": "Flutter",
        "localisable": False,
        "highlight": None,
        "definition": (
            "Fraction of total spectral power in the 5–12 Hz frequency band "
            "during expiration. Expiratory flutter can indicate upper airway "
            "instability persisting beyond the inspiratory phase. "
            "Amplitude-independent. Non-localisable: applies to the whole "
            "expiratory phase."
        ),
    },
    "flatness_insp_90": {
        "label": "Inspiratory Plateau (≥90% PIF)",
        "group": "Flatness",
        "localisable": True,
        "highlight": "insp_flatness90",  # shade samples ≥ 90% of PIF
        "definition": (
            "Fraction of the inspiratory phase where flow is at or above 90% of "
            "peak inspiratory flow (PIF). A high value means the breath has a "
            "flat plateau near its peak — characteristic of unobstructed flow. "
            "A low value (flow never sustains near its peak) suggests obstruction."
        ),
    },
    "insp_peak_position": {
        "label": "Inspiratory Peak Position",
        "group": "Asymmetry",
        "localisable": True,
        "highlight": "insp_peak",   # vertical marker at peak sample location
        "definition": (
            "Normalised position of peak inspiratory flow within the inspiratory "
            "phase (0 = very early, 1 = very late). In obstructed breaths, the "
            "peak often occurs earlier due to rapid onset and then limitation. "
            "Normal breaths tend to have a peak near the midpoint."
        ),
    },
    "insp_duty": {
        "label": "Inspiratory Duty Fraction",
        "group": "Timing",
        "localisable": True,
        "highlight": "insp_phase",   # bracket the full inspiratory phase
        "definition": (
            "Fraction of total breath duration spent in inspiration "
            "(inspiratory time ÷ total breath time). Flow-limited breaths "
            "tend to have a prolonged inspiratory phase relative to expiration, "
            "increasing this ratio."
        ),
    },
    "breath_duration": {
        "label": "Breath Duration",
        "group": "Timing",
        "localisable": True,
        "highlight": "full_breath",  # bracket the full breath
        "definition": (
            "Total duration of the breath window in seconds (end_s − start_s). "
            "Flow-limited breaths are often longer due to increased respiratory "
            "effort and slower, more effortful breathing patterns during sleep."
        ),
    },
    "exp_cv": {
        "label": "Expiratory Variability (CV)",
        "group": "Shape variability",
        "localisable": False,
        "highlight": None,
        "definition": (
            "Coefficient of variation of the expiratory flow signal (std ÷ mean), "
            "computed on the peak-normalised expiratory phase. Higher values "
            "indicate a more irregular expiratory waveform, which can occur due "
            "to expiratory flutter or turbulent flow associated with obstruction."
        ),
    },
    "insp_skew": {
        "label": "Inspiratory Skewness",
        "group": "Shape variability",
        "localisable": False,
        "highlight": None,
        "definition": (
            "Skewness of the normalised inspiratory flow distribution. Negative "
            "skew indicates the flow distribution has a longer left tail — flow "
            "spends more time at lower values (consistent with flattening or "
            "mid-inspiratory dip). Normal rounded breaths tend toward zero skew."
        ),
    },
    "insp_kurt": {
        "label": "Inspiratory Kurtosis",
        "group": "Shape variability",
        "localisable": False,
        "highlight": None,
        "definition": (
            "Excess kurtosis of the normalised inspiratory flow distribution. "
            "High kurtosis means the distribution is more peaked/heavy-tailed "
            "than a normal distribution — consistent with a sharply-peaked "
            "inspiratory waveform with rapid onset and fall, as seen in "
            "flow-limited breaths with early peak and subsequent collapse."
        ),
    },
}

# Ordered list matching training order (must match scaler + model column order)
FEATURE_NAMES = [
    "quad_insp_50",
    "area_under_peaks_insp",
    "power_5to12_insp",
    "power_5to12_exp",
    "flatness_insp_90",
    "insp_peak_position",
    "insp_duty",
    "breath_duration",
    "exp_cv",
    "insp_skew",
    "insp_kurt",
]

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

_model  = None
_scaler = None


def load_model():
    global _model, _scaler

    if not MODEL_PATH.exists():
        print(f"[WARN] Model not found at {MODEL_PATH}. Running in DEMO mode.")
        return
    if not SCALER_PATH.exists():
        print(f"[WARN] Scaler not found at {SCALER_PATH}. Running in DEMO mode.")
        return

    import joblib
    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    print(f"[INFO] LogReg model loaded from {MODEL_PATH}")
    print(f"[INFO] Scaler loaded from {SCALER_PATH}")


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# Mirrors 01-1_data_preparation_feateng.ipynb exactly.
# ─────────────────────────────────────────────

MIN_SAMPLES = 20
FS = 100.0


def _split_phases(flow: np.ndarray):
    insp = flow[flow > 0]
    exp  = -flow[flow < 0]
    return insp, exp


def _quadratic_deviation_mid50(insp_norm: np.ndarray) -> float:
    n = len(insp_norm)
    if n < 8:
        return np.nan
    q25, q75 = int(0.25 * n), int(0.75 * n)
    mid = insp_norm[q25:q75]
    t = np.linspace(0, 1, len(mid))
    parabola = 4 * t * (1 - t)
    return float(np.mean(np.abs(mid - parabola)))


def _band_power_ratio(phase: np.ndarray, fs: float,
                      fmin: float = 5.0, fmax: float = 12.0) -> float:
    if len(phase) < 20:
        return np.nan
    nperseg = min(32, len(phase) // 2)
    freqs, pxx = scipy_signal.welch(phase, fs=fs, nperseg=nperseg)
    total = np.sum(pxx)
    if total == 0:
        return np.nan
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(pxx[band_mask]) / total)


def _area_under_peaks(insp: np.ndarray) -> float:
    if len(insp) < 10:
        return np.nan
    peaks, _ = scipy_signal.find_peaks(insp, distance=max(3, len(insp) // 10))
    if len(peaks) < 2:
        return np.nan
    peak_line = np.interp(np.arange(len(insp)), peaks, insp[peaks])
    pif = np.max(insp)
    return float(np.mean(np.abs(insp - peak_line)) / (pif + 1e-9))


def _flatness(phase_norm: np.ndarray, threshold: float) -> float:
    return float(np.sum(phase_norm >= threshold) / len(phase_norm))


def extract_breath_features(flow: np.ndarray, duration_s: float,
                             fs: float = FS) -> dict | None:
    """
    Extract the 11-feature set matching model_logreg.pkl.
    Returns None if the breath is too short or has no valid phases.
    Also returns phase metadata needed for waveform highlighting.
    """
    if len(flow) < MIN_SAMPLES:
        return None

    insp, exp = _split_phases(flow)
    if len(insp) < 5 or len(exp) < 5:
        return None

    pif = float(np.max(insp))
    pef = float(np.max(exp))

    insp_norm = insp / (pif + 1e-9)
    exp_norm  = exp  / (pef + 1e-9)

    f = {}

    f["quad_insp_50"]          = _quadratic_deviation_mid50(insp_norm)
    f["area_under_peaks_insp"] = _area_under_peaks(insp)
    f["power_5to12_insp"]      = _band_power_ratio(insp, fs)
    f["power_5to12_exp"]       = _band_power_ratio(exp,  fs)
    f["flatness_insp_90"]      = _flatness(insp_norm, 0.90)
    f["insp_peak_position"]    = float(np.argmax(insp) / len(insp))
    f["insp_duty"]             = len(insp) / len(flow)
    f["breath_duration"]       = float(duration_s)
    f["exp_cv"]                = float(np.std(exp_norm) / (np.mean(exp_norm) + 1e-9))
    f["insp_skew"]             = float(skew(insp_norm))
    f["insp_kurt"]             = float(kurtosis(insp_norm))

    # ── Phase boundary metadata for waveform highlighting ────────────
    # Indices within the flow array; used by frontend to draw overlays.
    insp_indices = np.where(flow > 0)[0]
    exp_indices  = np.where(flow < 0)[0]

    insp_start_idx = int(insp_indices[0])  if len(insp_indices) > 0 else 0
    insp_end_idx   = int(insp_indices[-1]) if len(insp_indices) > 0 else len(flow)
    exp_start_idx  = int(exp_indices[0])   if len(exp_indices)  > 0 else insp_end_idx
    exp_end_idx    = int(exp_indices[-1])  if len(exp_indices)  > 0 else len(flow)

    n = len(flow)
    insp_mid25 = insp_start_idx + int(0.25 * (insp_end_idx - insp_start_idx))
    insp_mid75 = insp_start_idx + int(0.75 * (insp_end_idx - insp_start_idx))

    # Peak sample index within the full flow array
    insp_abs_indices = np.where(flow > 0)[0]
    if len(insp_abs_indices) > 0:
        peak_abs_idx = int(insp_abs_indices[np.argmax(insp)])
    else:
        peak_abs_idx = 0

    # Flatness: which samples are at >= 90% PIF
    flatness_mask = (flow >= 0.90 * pif).tolist()

    f["_phase_meta"] = {
        "n_samples":       n,
        "insp_start_idx":  insp_start_idx,
        "insp_end_idx":    insp_end_idx,
        "insp_mid25_idx":  insp_mid25,
        "insp_mid75_idx":  insp_mid75,
        "exp_start_idx":   exp_start_idx,
        "exp_end_idx":     exp_end_idx,
        "peak_abs_idx":    peak_abs_idx,
        "flatness_mask":   flatness_mask,  # bool list, one per sample
        "pif":             pif,
        "pef":             pef,
    }

    return f


# ─────────────────────────────────────────────
# PATIENT DATA LOADER
# ─────────────────────────────────────────────

def load_patient_folder(folder_path: Path) -> dict:
    participant_id = folder_path.name

    if not folder_path.exists():
        raise FileNotFoundError(f"Patient folder not found: {folder_path}")

    # Breath labels: prefer CSV, fall back to XLSX
    csv_files  = list(folder_path.glob("*breath*.csv"))
    xlsx_files = list(folder_path.glob("*breath*.xlsx"))

    if csv_files:
        breaths_df = pd.read_csv(csv_files[0])
    elif xlsx_files:
        breaths_df = pd.read_excel(xlsx_files[0])
    else:
        raise FileNotFoundError(f"No breath label file found in {folder_path}")

    breaths_df["participant"] = participant_id

    col_map = {}
    for col in breaths_df.columns:
        lc = col.lower().strip()
        if lc in ("start_s", "start (s)", "start(s)"):
            col_map[col] = "start_s"
        elif lc in ("end_s", "end (s)", "end(s)"):
            col_map[col] = "end_s"
        elif lc in ("breath #", "breath_number", "breath#"):
            col_map[col] = "breath_number"
    breaths_df = breaths_df.rename(columns=col_map)

    # Signal parquet
    meta_files    = list(folder_path.glob("*signals_meta.json"))
    parquet_files = list(folder_path.glob("*signals*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No signals parquet found in {folder_path}")

    with open(meta_files[0]) as f:
        meta = json.load(f)

    signals_df = pd.read_parquet(str(parquet_files[0]))
    signals_df["participant"] = participant_id

    flow_key = next(
        (k for k in signals_df.columns
         if "flow" in k.lower() and k not in ("participant",)),
        None
    )
    if flow_key is None:
        raise ValueError(f"No flow channel found in signals parquet for {participant_id}")

    # Match signals to breath windows
    combined_data = []
    for idx, row in breaths_df.iterrows():
        start_s = float(row.get("start_s", 0))
        end_s   = float(row.get("end_s",   0))
        label   = str(row.get("label", "NFL")).strip().upper()

        window = signals_df[
            (signals_df["time_s"] >= start_s) &
            (signals_df["time_s"] <  end_s)
        ].copy()

        if len(window) <= MIN_SAMPLES:
            continue

        combined_data.append({
            "participant":   participant_id,
            "breath_number": row.get("breath_number", idx),
            "label":         label,
            "target":        1 if label == "FL" else 0,
            "start_s":       start_s,
            "end_s":         end_s,
            "duration_s":    end_s - start_s,
            "flow":          window[flow_key].values.astype(float).tolist(),
            "time_s":        window["time_s"].values.tolist(),
        })

    if not combined_data:
        raise ValueError(f"No breath windows matched signals for {participant_id}")

    t_arr   = signals_df["time_s"].tolist()
    airflow = signals_df[flow_key].tolist()
    duration = float(signals_df["time_s"].max())

    breaths_out = [
        {"breath_id": i, "start_s": round(item["start_s"], 3), "end_s": round(item["end_s"], 3)}
        for i, item in enumerate(combined_data)
    ]

    return {
        "participant_id": participant_id,
        "breaths":        breaths_out,
        "combined_data":  combined_data,
        "airflow_t":      t_arr,
        "airflow_signal": airflow,
        "duration":       duration,
    }


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — ALL BREATHS
# ─────────────────────────────────────────────

def extract_all_features(patient_data: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns (features_df, phase_meta_list).
    features_df: shape (n_breaths, 11), columns = FEATURE_NAMES
    phase_meta_list: one dict per breath with phase boundary indices
    """
    rows      = []
    meta_list = []

    for item in patient_data["combined_data"]:
        flow     = np.array(item["flow"], dtype=float)
        duration = item["duration_s"]
        result   = extract_breath_features(flow, duration)

        if result is not None:
            phase_meta = result.pop("_phase_meta")
            rows.append({k: result.get(k, 0.0) for k in FEATURE_NAMES})
            meta_list.append(phase_meta)
        else:
            rows.append({k: 0.0 for k in FEATURE_NAMES})
            meta_list.append(None)

    df = pd.DataFrame(rows)[FEATURE_NAMES].fillna(0.0)
    return df, meta_list


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference(features_df: pd.DataFrame, patient_data: dict) -> List[dict]:
    breaths_out = []
    n = len(features_df)

    if _model is not None and _scaler is not None:
        X_scaled = _scaler.transform(features_df.values)
        proba    = _model.predict_proba(X_scaled)
        p_fl     = proba[:, -1]
    else:
        np.random.seed(42)
        p_fl = np.random.beta(1.5, 1.5, n)
        p_fl[:10]  = np.random.beta(8, 1, 10)
        p_fl[10:20] = np.random.beta(1, 8, 10)

    for i, (breath, p) in enumerate(zip(patient_data["breaths"], p_fl)):
        p = float(p)
        predicted_fl = p >= 0.5
        confidence   = p if predicted_fl else (1.0 - p)

        breaths_out.append({
            "breath_id":    breath["breath_id"],
            "start_s":      breath["start_s"],
            "end_s":        breath["end_s"],
            "prediction":   "FL" if predicted_fl else "NFL",
            "p_fl":         round(p, 4),
            "confidence":   round(confidence, 4),
            "needs_review": confidence < CONFIDENCE_THRESHOLD,
            "label_source": "model",
            "edited_at":    None,
            "edited_by":    None,
        })

    return breaths_out


# ─────────────────────────────────────────────
# LOGREG CONTRIBUTIONS
# Exact decomposition: contribution_i = coef_i × scaled_value_i
# Sum of all contributions + intercept = log-odds = logit(p_fl)
# ─────────────────────────────────────────────

def compute_logreg_contributions(features_df: pd.DataFrame,
                                  breath_idx: int,
                                  phase_meta: dict | None) -> list[dict]:
    """
    Returns list of dicts sorted by |contribution| descending.
    Each dict: {feature, raw_value, scaled_value, contribution, definition, localisable, highlight, label, group}
    """
    if _model is None or _scaler is None:
        # Demo fallback
        np.random.seed(breath_idx)
        demo_vals = np.random.randn(len(FEATURE_NAMES)) * 0.3
        return [
            {
                "feature":      feat,
                "raw_value":    round(float(features_df.iloc[breath_idx][feat]), 5),
                "scaled_value": round(float(demo_vals[i]), 5),
                "contribution": round(float(demo_vals[i] * 0.5), 5),
                **{k: FEATURE_DICT[feat][k]
                   for k in ("definition", "localisable", "highlight", "label", "group")},
            }
            for i, feat in enumerate(FEATURE_NAMES)
        ]

    row       = features_df.iloc[[breath_idx]]
    X_scaled  = _scaler.transform(row.values)  # shape (1, 11)
    coefs     = _model.coef_[0]                # shape (11,)
    raw_vals  = row.values[0]

    contributions = []
    for i, feat in enumerate(FEATURE_NAMES):
        contrib = float(coefs[i] * X_scaled[0, i])
        meta    = FEATURE_DICT.get(feat, {})
        contributions.append({
            "feature":      feat,
            "raw_value":    round(float(raw_vals[i]), 5),
            "scaled_value": round(float(X_scaled[0, i]), 5),
            "contribution": round(contrib, 5),
            "label":        meta.get("label", feat),
            "group":        meta.get("group", ""),
            "definition":   meta.get("definition", ""),
            "localisable":  meta.get("localisable", False),
            "highlight":    meta.get("highlight", None),
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    # Attach phase_meta to localisable features so frontend can render overlays
    if phase_meta:
        for c in contributions:
            if c["localisable"]:
                c["phase_meta"] = phase_meta

    return contributions


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

sessions: dict[str, dict] = {}


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "mode": "DEMO" if _model is None else "LIVE",
        "model": str(MODEL_PATH),
        "features": FEATURE_NAMES,
    }


@app.get("/api/feature_dict")
def get_feature_dict():
    """
    Return the full FEATURE_DICT so the frontend can display
    definitions and localisability without hardcoding them.
    """
    return FEATURE_DICT


@app.post("/api/session/upload")
async def upload_patient_folder(patient_id: str):
    folder_path = PROCESSED_DATA_DIR / patient_id

    try:
        patient_data              = load_patient_folder(folder_path)
        features_df, phase_metas  = extract_all_features(patient_data)
        breath_results            = run_inference(features_df, patient_data)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    session_id = f"{patient_id}_{int(time.time())}"
    sessions[session_id] = {
        "patient_id":   patient_id,
        "created_at":   datetime.utcnow().isoformat(),
        "patient_data": patient_data,
        "features_df":  features_df,
        "phase_metas":  phase_metas,
        "breaths":      breath_results,
    }

    n_total  = len(breath_results)
    n_fl     = sum(1 for b in breath_results if b["prediction"] == "FL")
    n_review = sum(1 for b in breath_results if b["needs_review"])

    return {
        "session_id":           session_id,
        "patient_id":           patient_id,
        "n_breaths":            n_total,
        "n_fl":                 n_fl,
        "n_nfl":                n_total - n_fl,
        "n_needs_review":       n_review,
        "fl_percent":           round(100 * n_fl / n_total, 1) if n_total else 0,
        "breaths":              breath_results,
        "airflow_t":            patient_data["airflow_t"],
        "airflow_signal":       patient_data["airflow_signal"],
        "duration":             patient_data["duration"],
        "model_mode":           "DEMO" if _model is None else "LIVE",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/api/session/{session_id}/explain/{breath_id}")
async def explain_breath(session_id: str, breath_id: int):
    """
    Return exact LogReg per-feature contributions for a single breath.
    Auto-called by frontend on breath selection.

    Response includes:
      - contributions: list sorted by |contribution| desc
      - intercept: model intercept (for log-odds reconstruction)
      - log_odds: sum(contributions) + intercept
      - p_fl: sigmoid(log_odds)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess        = sessions[session_id]
    features_df = sess["features_df"]
    phase_metas = sess["phase_metas"]

    if breath_id < 0 or breath_id >= len(features_df):
        raise HTTPException(status_code=404, detail="Breath ID out of range")

    phase_meta   = phase_metas[breath_id] if phase_metas else None
    contributions = compute_logreg_contributions(features_df, breath_id, phase_meta)

    # Reconstruct log-odds for verification
    total_contrib = sum(c["contribution"] for c in contributions)
    intercept     = float(_model.intercept_[0]) if _model is not None else 0.0
    log_odds      = total_contrib + intercept
    p_fl_check    = float(1 / (1 + np.exp(-log_odds)))

    return {
        "breath_id":     breath_id,
        "contributions": contributions,
        "intercept":     round(intercept, 5),
        "log_odds":      round(log_odds, 5),
        "p_fl_check":    round(p_fl_check, 4),
    }


@app.patch("/api/session/{session_id}/breath/{breath_id}")
async def update_breath_label(session_id: str, breath_id: int, payload: dict):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    breaths = sessions[session_id]["breaths"]
    breath  = next((b for b in breaths if b["breath_id"] == breath_id), None)
    if breath is None:
        raise HTTPException(status_code=404, detail="Breath not found")

    new_label = payload.get("label", "").upper()
    if new_label not in ("FL", "NFL"):
        raise HTTPException(status_code=400, detail="label must be FL or NFL")

    breath["prediction"]   = new_label
    breath["label_source"] = "user"
    breath["edited_at"]    = datetime.utcnow().isoformat()
    breath["edited_by"]    = payload.get("edited_by", "unknown")
    breath["needs_review"] = False

    return breath


@app.post("/api/session/{session_id}/save")
async def save_session(session_id: str, payload: dict):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess      = sessions[session_id]
    breaths   = sess["breaths"]
    patient_id = sess["patient_id"]

    rows = [{
        "patient_id":   patient_id,
        "breath_id":    b["breath_id"],
        "start_s":      b["start_s"],
        "end_s":        b["end_s"],
        "label":        b["prediction"],
        "p_fl":         b["p_fl"],
        "confidence":   b["confidence"],
        "label_source": b["label_source"],
        "edited_at":    b["edited_at"],
        "edited_by":    b["edited_by"],
        "saved_at":     datetime.utcnow().isoformat(),
        "saved_by":     payload.get("saved_by", "unknown"),
    } for b in breaths]

    df       = pd.DataFrame(rows)
    out_path = SESSION_OUTPUT_DIR / f"{patient_id}_{session_id}.csv"
    df.to_csv(out_path, index=False)

    return {
        "saved":         True,
        "path":          str(out_path),
        "n_breaths":     len(rows),
        "n_user_edited": sum(1 for r in rows if r["label_source"] == "user"),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
