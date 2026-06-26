"""
Meme Kanseri NAC Yanıt Tahmini — Ana Pipeline

Adımlar (sırayla):
  1. data_loader  →  DICOM katalog + klinik tablo birleştirme
  2. harmonize    →  DCE-MRI ön işleme ve ROI patch çıkarma
  3. ml_models    →  5 klasik ML modeli (GroupKFold CV)
  4. cnn_model    →  EfficientNet-B0 transfer learning
  5. shap_gradcam →  SHAP + GradCAM++ açıklanabilirlik
  6. visualization →  tüm grafikler

Kullanım:
  python main.py                          # tam pipeline
  python main.py --steps ml viz           # seçili adımlar
  python main.py --demo                   # gerçek veri olmadan demo modu
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Proje yollarını ekle ──────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    BASE_DIR, CLINICAL_DIR, CV_FOLDS, MODELS_DIR,
    PLOTS_DIR, PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

# ── Adım sonuç kaydı ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    name:     str
    status:   str = "pending"   # pending | ok | skipped | failed
    elapsed:  float = 0.0
    error:    str = ""
    outputs:  dict = field(default_factory=dict)


# ── Pipeline durumu ───────────────────────────────────────────────────────────

class PipelineState:
    """Adımlar arasında paylaşılan veri."""
    def __init__(self):
        self.scan_df:      pd.DataFrame | None = None
        self.clinical_df:  pd.DataFrame | None = None
        self.merged_df:    pd.DataFrame | None = None
        self.labels_df:    pd.DataFrame | None = None

        # ML adımı çıktıları
        self.fold_df:      pd.DataFrame | None = None
        self.cv_summary:   dict | None = None
        self.model_preds:  dict | None = None   # {name: (y_true, y_prob)}
        self.pipelines:    dict | None = None   # {name: fitted Pipeline}
        self.thresholds:   dict | None = None

        # CNN adımı çıktıları
        self.cnn_results:  dict | None = None

        # Demo modu bayrağı
        self.demo: bool = False


# ── model_preds disk yardımcıları ────────────────────────────────────────────

_MODEL_PREDS_NPZ  = RESULTS_DIR / "model_preds.npz"
_THRESHOLDS_JSON  = RESULTS_DIR / "model_thresholds.json"


def _save_model_preds(model_preds: dict, thresholds: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (y_true, y_prob) in model_preds.items():
        arrays[f"{name}__y_true"] = np.asarray(y_true)
        arrays[f"{name}__y_prob"] = np.asarray(y_prob)
    np.savez(_MODEL_PREDS_NPZ, **arrays)
    _THRESHOLDS_JSON.write_text(json.dumps(thresholds), encoding="utf-8")
    log.info("model_preds diskе kaydedildi: %s", _MODEL_PREDS_NPZ)


def _load_model_preds() -> tuple[dict, dict]:
    if not _MODEL_PREDS_NPZ.exists():
        return {}, {}
    data = np.load(str(_MODEL_PREDS_NPZ))
    model_preds: dict = {}
    for key in data.files:
        if key.endswith("__y_true"):
            name = key[: -len("__y_true")]
            model_preds[name] = (data[f"{name}__y_true"], data[f"{name}__y_prob"])
    thresholds: dict = {}
    if _THRESHOLDS_JSON.exists():
        thresholds = json.loads(_THRESHOLDS_JSON.read_text(encoding="utf-8"))
    log.info("model_preds diskten yuklendi: %d model", len(model_preds))
    return model_preds, thresholds


# ── Adım çalıştırıcı ─────────────────────────────────────────────────────────

def _run_step(name: str, fn, *args, **kwargs) -> tuple[StepResult, object]:
    """
    fn(*args, **kwargs)'i zamanlayarak çalıştırır.
    Hata durumunda None döner ve StepResult.status = 'failed' olur.
    """
    log.info("")
    log.info("=" * 60)
    log.info("ADIM: %s", name.upper())
    log.info("=" * 60)

    result = StepResult(name=name, status="running")
    t0 = time.perf_counter()
    output = None

    try:
        output = fn(*args, **kwargs)
        result.status  = "ok"
        result.elapsed = time.perf_counter() - t0
        log.info("[%s] TAMAMLANDI (%.1fs)", name, result.elapsed)
    except Exception as exc:
        result.status  = "failed"
        result.elapsed = time.perf_counter() - t0
        result.error   = f"{type(exc).__name__}: {exc}"
        log.error("[%s] HATA — %s", name, result.error)
        log.debug(traceback.format_exc())

    return result, output


# ── Adım 1: data_loader ───────────────────────────────────────────────────────

def step_data_loader(state: PipelineState) -> dict:
    from data_loader import scan_ispy1_patients, load_clinical, merge_image_clinical

    scan_df     = scan_ispy1_patients(dce_only=True, t0_only=True)
    clinical_df = load_clinical()
    merged_df, summary = merge_image_clinical(scan_df, clinical_df)

    state.scan_df     = scan_df
    state.clinical_df = clinical_df
    state.merged_df   = merged_df
    state.labels_df   = merged_df[["patient_id", "pCR"]].dropna()

    # Birleşik katalogu kaydet
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "merged_catalog.csv"
    merged_df.to_csv(out, index=False)
    log.info("Birlesik katalog: %s  (%d hasta)", out, len(merged_df))

    return {
        "n_patients":     len(merged_df),
        "pCR_positive":   summary.get("pCR_positive"),
        "pCR_negative":   summary.get("pCR_negative"),
        "catalog_path":   str(out),
    }


# ── Adım 2: harmonize ─────────────────────────────────────────────────────────

def step_harmonize(state: PipelineState) -> dict:
    from harmonize import process_all_patients

    results = process_all_patients()
    n_ok    = len(results["ok"])
    n_fail  = len(results["failed"])
    log.info("Harmonizasyon: %d OK, %d hatali", n_ok, n_fail)
    return {"processed_ok": n_ok, "processed_failed": n_fail}


# ── NACT-Pilot T1 delta özellik hesaplama ─────────────────────────────────────

def _compute_nact_pilot_delta_features(
    radiomic_csv: Path,
    raw_nact_dir: Path,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    NACT-Pilot hastalarından (numeric ID 1-68) T1-T0 delta özelliklerini hesaplar.

    NACT-Pilot DICOM yapısı:
      raw_nact_dir/.../Breast-MRI-NACT-Pilot/UCSF-BR-XX/
        <T0-tarih>/   ← en erken, zaten işlenmiş
        <T1-tarih>/   ← ikinci tarih, bu fonksiyon burayı işler

    Hesaplanan delta sütunları:
      delta_ser     = kin_ser_mean(T1) − kin_ser_mean(T0)
      delta_ftv     = kin_tumor_volume(T1) − kin_tumor_volume(T0)
      delta_ce_slope = kin_ce_slope(T1) − kin_ce_slope(T0)

    Sadece T1 verisi başarıyla işlenen hastalar için değer döner;
    diğerleri NaN kalır (sonraki medyan dolgusu devreye girer).

    Returns
    -------
    pd.DataFrame — sütunlar: patient_id, delta_ser, delta_ftv, delta_ce_slope
    """
    import csv as _csv

    delta_cols = ["delta_ser", "delta_ftv", "delta_ce_slope"]
    records: list[dict] = []

    # ── Mevcut T0 radyomik verisi ──────────────────────────────────────────────
    if not radiomic_csv.exists():
        log.warning("Radyomik CSV bulunamadi; delta özellikleri atlanıyor.")
        return pd.DataFrame(columns=["patient_id"] + delta_cols)

    t0_df = pd.read_csv(radiomic_csv)
    t0_df["patient_id"] = t0_df["patient_id"].astype(str).str.strip()
    # NACT-Pilot hastalari: numerik ID 1-68
    nact_ids = {
        r["patient_id"]
        for _, r in t0_df.iterrows()
        if r["patient_id"].isdigit() and 1 <= int(r["patient_id"]) <= 68
    }
    if not nact_ids:
        log.info("Radyomik CSV'de NACT-Pilot hastalari bulunamadi (ID 1-68).")
        return pd.DataFrame(columns=["patient_id"] + delta_cols)

    # ── NACT-Pilot ham dizin → UCSF-BR-XX alt klasörleri ─────────────────────
    nact_root = None
    for candidate in raw_nact_dir.rglob("Breast-MRI-NACT-Pilot"):
        if candidate.is_dir():
            nact_root = candidate
            break
    if nact_root is None:
        log.warning("Breast-MRI-NACT-Pilot dizini bulunamadi: %s", raw_nact_dir)
        return pd.DataFrame(columns=["patient_id"] + delta_cols)

    # ── Harmonize + radyomik araçlarını içe aktar ─────────────────────────────
    try:
        from harmonize import (
            discover_dce_phases, process_patient,
            normalize_volume, compute_ser, build_rgb_patch,
            find_tumor_center,
        )
        import SimpleITK as sitk
        from radiomic_features import extract_all_features
    except ImportError as exc:
        log.warning("Harmonize/radiomic modülü içe aktarilamadi: %s — delta atlanıyor.", exc)
        return pd.DataFrame(columns=["patient_id"] + delta_cols)

    # ── Her NACT-Pilot hastası için T1 işle ──────────────────────────────────
    for ucsf_dir in sorted(nact_root.iterdir()):
        if not ucsf_dir.is_dir():
            continue
        # UCSF-BR-XX → numeric ID
        m = re.match(r"^UCSF-BR-0*(\d+)$", ucsf_dir.name, re.I)
        if not m:
            continue
        pid = m.group(1).lstrip("0") or "0"
        if pid not in nact_ids:
            continue  # bu hasta radyomik CSV'de yok

        # Çalışma tarihlerini sırala (T0=en erken, T1=ikinci)
        study_dirs = sorted([d for d in ucsf_dir.iterdir() if d.is_dir()])
        if len(study_dirs) < 2:
            log.debug("[%s] T1 çalışması yok; delta atlanıyor.", pid)
            continue
        t1_study = study_dirs[1]   # ikinci tarih = T1

        # ── T1 patch'ini geçici bir dizine çıkar ─────────────────────────────
        tmp_dir = processed_dir / "_t1_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        t1_npy = tmp_dir / f"{pid}_T1.npy"

        try:
            if not t1_npy.exists():
                # Sadece T1 study'sini işle — process_patient yerine adım adım
                from harmonize import (
                    _load_sitk_series, reorient_to_axial_lps,
                    resample_to_spacing, _get_field_strength,
                    _tissue_mask, _match_shape,
                )
                # Faz keşfi — tek bir study dir'i wrap'le
                class _SingleStudyWrapper:
                    def __init__(self, d): self._d = d
                    def iterdir(self): return [self._d]
                    def __truediv__(self, k): return self._d / k

                tmp_patient = _SingleStudyWrapper(t1_study)
                # discover_dce_phases patient_dir.iterdir() kullandığından
                # geçici bir wrapper dir oluşturuyoruz
                import tempfile, os
                with tempfile.TemporaryDirectory() as td:
                    link = Path(td) / "t1_study"
                    # Sembolik link veya hard link yerine discover'ı doğrudan çağır
                    # discover_dce_phases'ı tek study dir ile çalıştırıyoruz
                    orig_iterdir = type(ucsf_dir).iterdir

                    class _FakePatientDir:
                        """discover_dce_phases'ın sadece T1 study'yi görmesini sağlar."""
                        def __init__(self, study_d): self._study = study_d
                        def iterdir(self): return iter([self._study])

                    fake_patient = _FakePatientDir(t1_study)
                    phase_set = discover_dce_phases(fake_patient)  # type: ignore[arg-type]

                if not phase_set.is_complete():
                    log.debug("[%s] T1 study DCE fazlari eksik; atlanıyor.", pid)
                    continue

                # Yükle, normalize et, patch çıkar
                def _lo(sd):
                    img = _load_sitk_series(sd)
                    img = reorient_to_axial_lps(img)
                    return img

                pre_img   = _lo(phase_set.pre)
                early_img = _lo(phase_set.early)
                late_img  = _lo(phase_set.late)

                fs = _get_field_strength(phase_set.pre)
                for im in [pre_img, early_img, late_img]:
                    im = resample_to_spacing(im)

                pre_img   = resample_to_spacing(pre_img)
                early_img = resample_to_spacing(early_img)
                late_img  = resample_to_spacing(late_img)

                pre_v   = sitk.GetArrayFromImage(pre_img).astype(np.float32)
                early_v = sitk.GetArrayFromImage(early_img).astype(np.float32)
                late_v  = sitk.GetArrayFromImage(late_img).astype(np.float32)
                early_v = _match_shape(early_v, pre_v.shape)
                late_v  = _match_shape(late_v,  pre_v.shape)

                mask  = _tissue_mask(pre_v)
                pre_v   = normalize_volume(pre_v,   fs, mask)
                early_v = normalize_volume(early_v, fs, mask)
                late_v  = normalize_volume(late_v,  fs, mask)

                ser    = compute_ser(pre_v, early_v, late_v)
                center = find_tumor_center(ser)
                patch  = build_rgb_patch(pre_v, early_v, late_v, center, 128)
                np.save(str(t1_npy), patch)
            else:
                patch = np.load(str(t1_npy)).astype(np.float32)

            # ── T1 radyomik özellikler ────────────────────────────────────────
            t1_feats = extract_all_features(patch, pid)

            # ── T0 radyomik değerler ──────────────────────────────────────────
            t0_row = t0_df[t0_df["patient_id"] == pid]
            if t0_row.empty:
                continue
            t0 = t0_row.iloc[0]

            records.append({
                "patient_id":    pid,
                "delta_ser":     t1_feats.get("kin_ser_mean",      np.nan) - float(t0.get("kin_ser_mean",      np.nan)),
                "delta_ftv":     t1_feats.get("kin_tumor_volume",  np.nan) - float(t0.get("kin_tumor_volume",  np.nan)),
                "delta_ce_slope": t1_feats.get("kin_ce_slope",    np.nan) - float(t0.get("kin_ce_slope",     np.nan)),
            })
            log.info("[%s] T1 delta hesaplandi: Δser=%.3f Δftv=%.3f Δce=%.3f",
                     pid,
                     records[-1]["delta_ser"],
                     records[-1]["delta_ftv"],
                     records[-1]["delta_ce_slope"])

        except Exception as exc:
            log.warning("[%s] T1 isleme hatasi: %s", pid, exc)

    if records:
        log.info("NACT-Pilot delta ozellikler: %d hasta islendi.", len(records))
    else:
        log.info("NACT-Pilot T1 isleme tamamlanamadi veya veri yok; delta atlaniyor.")

    return pd.DataFrame(records, columns=["patient_id"] + delta_cols) if records \
           else pd.DataFrame(columns=["patient_id"] + delta_cols)


# ── Adım 3: ml_models ─────────────────────────────────────────────────────────

def step_ml_models(state: PipelineState, use_smote: bool = True) -> dict:
    from ml_models import (
        run_all_models, summarize_results, save_results,
        build_pipeline, MODEL_NAMES,
    )

    labels_df = state.labels_df
    if labels_df is None or labels_df.empty:
        raise ValueError("labels_df bos — once data_loader adimini calistirin.")

    clinical_df = state.clinical_df
    CLINICAL_FEATURES = ["ER", "PR", "HER2", "age", "tumor_size", "grade"]

    # ── Klinik özellikler ──────────────────────────────────────────────────────
    feat_cols = [c for c in CLINICAL_FEATURES
                 if c in (clinical_df if clinical_df is not None else pd.DataFrame()).columns]

    _pos = re.compile(r"^(pos|yes|1|true|amplified|equivocal)", re.I)
    _neg = re.compile(r"^(neg|no|0|false|not.amplified)", re.I)

    def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: (
                    1 if pd.notna(v) and _pos.match(str(v)) else
                    0 if pd.notna(v) and _neg.match(str(v)) else
                    np.nan
                ))
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.fillna(df.median(numeric_only=True)).fillna(0)
        return df

    if feat_cols and state.merged_df is not None:
        X_clin = _encode_df(state.merged_df[feat_cols])
    else:
        X_clin = pd.DataFrame(index=labels_df.index)

    # ── Radyomik özellikler (varsa) ────────────────────────────────────────────
    radiomic_path = RESULTS_DIR / "radiomic_features.csv"
    X_rad = pd.DataFrame()
    if radiomic_path.exists():
        try:
            rad_df = pd.read_csv(radiomic_path, dtype=str)
            rad_df["patient_id"] = rad_df["patient_id"].astype(str).str.strip()

            # Sadece sayısal radyomik sütunları al
            _RAD_PREFIXES = ("int_", "kin_", "glcm_", "lbp_", "shape_")
            rad_cols = [c for c in rad_df.columns
                        if any(c.startswith(p) for p in _RAD_PREFIXES)]
            rad_sub = rad_df[["patient_id"] + rad_cols].copy()

            # labels_df ile hizala
            labels_with_pid = labels_df.reset_index()
            merged_rad = pd.merge(
                labels_with_pid[["index", "patient_id"]],
                rad_sub,
                on="patient_id", how="left",
            ).set_index("index")
            merged_rad = merged_rad.drop(columns=["patient_id"], errors="ignore")
            X_rad = _encode_df(merged_rad)
            log.info("Radyomik ozellikler yuklendi: %d sutun", X_rad.shape[1])
        except Exception as exc:
            log.warning("Radyomik ozellik dosyasi okunamadi: %s", exc)

    # ── Özellik matrisi birleştir ──────────────────────────────────────────────
    frames = [f for f in [X_clin, X_rad] if not f.empty]
    if frames:
        X = pd.concat(frames, axis=1)
        # Tekrarlayan sütunları kaldır
        X = X.loc[:, ~X.columns.duplicated()]
        log.info("Toplam ozellik matrisi: %d hasta x %d ozellik", X.shape[0], X.shape[1])
    else:
        log.warning("Hic ozellik yok — sentetik ozellikler kullaniliyor.")
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(labels_df)
        X = pd.DataFrame(
            rng.standard_normal((n, 10)),
            columns=[f"synthetic_{i}" for i in range(10)],
            index=labels_df.index,
        )

    y       = labels_df["pCR"].astype(int)
    groups  = labels_df["patient_id"]

    # Satır hizalaması
    common_idx = X.index.intersection(y.index)
    X, y, groups = X.loc[common_idx], y.loc[common_idx], groups.loc[common_idx]

    # ── NACT-Pilot T1 delta özellikler — özellik seçiminden ÖNCE eklenir ────────
    raw_nact = BASE_DIR / "data" / "raw" / "NACT-Pilot"
    delta_df = _compute_nact_pilot_delta_features(
        radiomic_csv=radiomic_path,
        raw_nact_dir=raw_nact,
        processed_dir=PROCESSED_DIR,
    )
    if not delta_df.empty:
        delta_df["patient_id"] = delta_df["patient_id"].astype(str).str.strip()
        pid_to_idx = {
            str(pid).strip(): idx
            for idx, pid in zip(labels_df.loc[common_idx, "patient_id"], common_idx)
        }
        delta_indexed = pd.DataFrame(
            np.nan, index=X.index,
            columns=["delta_ser", "delta_ftv", "delta_ce_slope"],
        )
        for _, dr in delta_df.iterrows():
            row_idx = pid_to_idx.get(str(dr["patient_id"]))
            if row_idx is not None and row_idx in delta_indexed.index:
                for col in ["delta_ser", "delta_ftv", "delta_ce_slope"]:
                    delta_indexed.at[row_idx, col] = dr[col]
        delta_indexed = delta_indexed.fillna(delta_indexed.median(numeric_only=True)).fillna(0)
        n_ok = int((~delta_indexed["delta_ser"].eq(delta_indexed["delta_ser"].median())).sum())
        X = pd.concat([X, delta_indexed], axis=1)
        log.info("Delta özellikler X'e eklendi: %d NACT-Pilot hastası, +3 sütun.", n_ok)

    # ── 3D Radyomik özellikler ────────────────────────────────────────────────────
    rad3d_path = RESULTS_DIR / "radiomic_features_3d.csv"
    try:
        from radiomic_features_3d import extract_3d_features_all_patients
        ispy1_pids = [str(p) for p in labels_df.loc[common_idx, "patient_id"]
                      if int(str(p).strip()) >= 1000]
        rad3d_df = extract_3d_features_all_patients(
            processed_dir=PROCESSED_DIR,
            output_path=rad3d_path,
            patient_ids=ispy1_pids if ispy1_pids else None,
        )
        if rad3d_df is not None and not rad3d_df.empty:
            rad3d_df["patient_id"] = rad3d_df["patient_id"].astype(str).str.strip()
            _rad3d_cols = [c for c in rad3d_df.columns if c != "patient_id"]
            labels_with_pid2 = labels_df.reset_index()
            merged_3d = pd.merge(
                labels_with_pid2[["index", "patient_id"]],
                rad3d_df[["patient_id"] + _rad3d_cols],
                on="patient_id", how="left",
            ).set_index("index")
            merged_3d = merged_3d.drop(columns=["patient_id"], errors="ignore")
            merged_3d = _encode_df(merged_3d.loc[X.index])
            X = pd.concat([X, merged_3d], axis=1)
            X = X.loc[:, ~X.columns.duplicated()]
            log.info("3D radyomik özellikler eklendi: +%d sütun", merged_3d.shape[1])
    except Exception as exc:
        log.warning("3D radyomik özellikleri yüklenemedi: %s", exc)

    # ── İnteraksiyon terimleri ────────────────────────────────────────────────────
    _ia_cols = list(X.columns)
    def _safe_col(name):
        return X[name] if name in _ia_cols else pd.Series(0.0, index=X.index)

    er  = _safe_col("ER")
    pr  = _safe_col("PR")
    her2 = _safe_col("HER2")
    vol  = _safe_col("kin_tumor_volume")
    wash = _safe_col("kin_washout_ratio")

    X = X.copy()
    X["IA_ER_x_HER2"]     = er  * her2
    X["IA_PR_x_HER2"]     = pr  * her2
    X["IA_ER_x_PR"]       = er  * pr
    X["IA_vol_x_washout"] = vol * wash
    log.info("İnteraksiyon terimleri eklendi: ER×HER2, PR×HER2, ER×PR, vol×washout")

    # ── LASSO + Mutual Information birleşik özellik seçimi ───────────────────────
    N_TOP = 25
    if X.shape[1] > N_TOP:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # LASSO katsayı büyüklükleri
        lasso = LassoCV(
            cv=min(5, int((y == 1).sum())),
            max_iter=10000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        lasso.fit(X_sc, y)
        coef_abs = np.abs(lasso.coef_)

        # Mutual Information skorları
        mi_scores = mutual_info_classif(X_sc, y, random_state=RANDOM_SEED, n_neighbors=3)

        # Her iki ölçütü sıra (rank) normalize edip eşit ağırlıkla birleştir
        def _rank_norm(arr: np.ndarray) -> np.ndarray:
            ranks = arr.argsort().argsort().astype(float)
            return ranks / max(ranks.max(), 1.0)

        combined = _rank_norm(coef_abs) + _rank_norm(mi_scores)
        top_idx  = np.argsort(combined)[::-1][:N_TOP]
        top_cols = X.columns[top_idx].tolist()

        log.info(
            "LASSO+MI secimi: %d → top-%d | LASSO nonzero=%d | MI max=%.4f | secilen: %s...",
            X.shape[1], N_TOP,
            int((coef_abs > 0).sum()), float(mi_scores.max()), top_cols[:5],
        )
        X = X[top_cols]
    else:
        log.info("Ozellik sayisi (%d) <= %d — secim atlandi.", X.shape[1], N_TOP)

    # ── "Öncesi" metrikleri kaydet (karşılaştırma için) ───────────────────────
    _prev_summary_path = RESULTS_DIR / "ml_results_summary.json"
    prev_summary: dict = {}
    if _prev_summary_path.exists():
        try:
            prev_summary = json.loads(_prev_summary_path.read_text(encoding="utf-8"))
            log.info("Önceki özet yüklendi (before/after karşılaştırması için).")
        except Exception:
            pass

    fold_df, oof_by_model = run_all_models(X, y, groups=groups, use_smote=use_smote)
    cv_summary = summarize_results(fold_df, oof_by_model)
    csv_p, json_p = save_results(fold_df, cv_summary)

    # ── OOF tahminlerini topla (görselleştirme için) ──────────────────────────
    model_preds: dict[str, tuple] = {}
    thresholds:  dict[str, float] = {}

    from ml_models import f1_threshold as _f1_thr
    for model_name in MODEL_NAMES:
        pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
        pipe = build_pipeline(model_name, pos_weight)
        pipe.fit(X, y)
        y_prob = pipe.predict_proba(X)[:, 1]
        model_preds[model_name] = (y.values, y_prob)
        # F1-maximizing threshold (OOF-dan)
        if model_name in oof_by_model:
            oof = oof_by_model[model_name]
            thresholds[model_name] = _f1_thr(oof["y_true"], oof["y_prob"])
        else:
            thresholds[model_name] = _f1_thr(y.values, y_prob)

    state.fold_df     = fold_df
    state.cv_summary  = cv_summary
    state.model_preds = model_preds
    state.thresholds  = thresholds
    state.pipelines   = {}   # kayıtlı modeller opsiyonel

    # model_preds ve thresholds'u diske kaydet (viz adımı tek başına çalışabilsin)
    _save_model_preds(model_preds, thresholds)

    from ml_models import print_summary_table
    print_summary_table(cv_summary)

    # ── Before / After karşılaştırması ───────────────────────────────────────
    if prev_summary:
        _COMPARE_METRICS = ["auc_roc", "f1", "sensitivity", "mcc"]
        _COMPARE_MODELS  = ["ensemble", "random_forest", "lightgbm", "xgboost"]
        sep = "=" * 70
        print(f"\n{sep}")
        print("  BEFORE → AFTER KARŞILAŞTIRMASI  (Δ = yeni − eski)")
        print(sep)
        hdr = f"{'Model':<16} " + " ".join(f"{m.upper():>12}" for m in _COMPARE_METRICS)
        print(hdr)
        print("-" * len(hdr))
        for mn in _COMPARE_MODELS:
            if mn not in cv_summary or mn not in prev_summary:
                continue
            row = f"{mn:<16}"
            for m in _COMPARE_METRICS:
                new_val  = cv_summary[mn].get(m, {}).get("mean", float("nan"))
                old_val  = prev_summary[mn].get(m, {}).get("mean", float("nan"))
                if not (np.isnan(new_val) or np.isnan(old_val)):
                    delta    = new_val - old_val
                    sign     = "+" if delta >= 0 else ""
                    row     += f"  {old_val:.3f}→{new_val:.3f}({sign}{delta:.3f})"
                else:
                    row     += f"  {'?':>12}"
            print(row)
        # Bootstrap CI özeti
        print(f"\n  Bootstrap 95% CI (AUC, n=1000):")
        for mn in _COMPARE_MODELS:
            ci = cv_summary.get(mn, {}).get("auc_bootstrap", {})
            if ci:
                print(f"    {mn:<16} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
                      f"  mean={ci['mean']:.3f}")
        print(sep + "\n")
    else:
        log.info("Önceki özet bulunamadi; before/after karşılaştırması atlanıyor.")

    return {"csv": str(csv_p), "json": str(json_p), "n_models": len(MODEL_NAMES)}


# ── Adım 4: cnn_model ─────────────────────────────────────────────────────────

def step_cnn(state: PipelineState) -> dict:
    import torch
    from cnn_model import run_cv_training, save_cnn_results

    labels_df = state.labels_df
    if labels_df is None or labels_df.empty:
        raise ValueError("labels_df bos.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("CNN egitim cihazi: %s", device)

    try:
        cv_out = run_cv_training(
            labels_df,
            patch_dir=PROCESSED_DIR,
            n_splits=CV_FOLDS,
            device=device,
        )
    except FileNotFoundError as exc:
        if state.demo:
            log.warning("CNN adimi atlaniyor (demo modunda patch yok): %s", exc)
            return {"skipped": True, "reason": "demo_no_patches"}
        raise

    results_path = save_cnn_results(cv_out)
    state.cnn_results = cv_out

    # CNN tahminlerini model_preds'e ekle (görselleştirme için)
    best_ckpt = MODELS_DIR / "best_cnn.pt"
    if best_ckpt.exists() and state.model_preds is not None:
        try:
            from cnn_model import load_best_model, PatchDataset
            from torch.utils.data import DataLoader

            cnn_model = load_best_model(device)
            ds  = PatchDataset(PROCESSED_DIR, labels_df, train=False)
            dl  = DataLoader(ds, batch_size=16, shuffle=False)
            all_probs, all_labels = [], []
            with torch.no_grad():
                for imgs, lbls in dl:
                    logits = cnn_model(imgs.to(device)).squeeze(1)
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    all_labels.extend(lbls.numpy())
            state.model_preds["cnn"] = (
                np.array(all_labels), np.array(all_probs)
            )
            _save_model_preds(state.model_preds, state.thresholds or {})
        except Exception as exc:
            log.warning("CNN tahmini model_preds'e eklenemedi: %s", exc)

    summary = cv_out.get("summary", {})
    auc_mean = summary.get("auc_roc", {}).get("mean", "?")
    log.info("CNN CV AUC: %s", auc_mean)
    return {"results_path": str(results_path), "auc_mean": auc_mean}


# ── Adım 5: shap_gradcam ──────────────────────────────────────────────────────

def step_shap_gradcam(state: PipelineState) -> dict:
    import torch
    from shap_gradcam import (
        run_shap_analysis, shap_waterfall,
        run_gradcam_batch, save_dice_report,
    )
    from ml_models import build_pipeline
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    outputs: dict = {}

    labels_df = state.labels_df
    if labels_df is None or labels_df.empty:
        log.warning("labels_df bos; SHAP/GradCAM adimi atlaniyor.")
        return outputs

    # ── Klinik + radyomik özellik matrisi (step_ml_models ile aynı) ────────────
    CLINICAL_FEATURES = ["ER", "PR", "HER2", "age", "tumor_size", "grade"]
    clinical_df = state.clinical_df
    feat_cols = [c for c in CLINICAL_FEATURES
                 if c in (clinical_df if clinical_df is not None else pd.DataFrame()).columns]

    _pos = re.compile(r"^(pos|yes|1|true|amplified|equivocal)", re.I)
    _neg = re.compile(r"^(neg|no|0|false|not.amplified)", re.I)

    def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: (
                    1 if pd.notna(v) and _pos.match(str(v)) else
                    0 if pd.notna(v) and _neg.match(str(v)) else
                    np.nan
                ))
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.fillna(df.median(numeric_only=True)).fillna(0)
        return df

    if feat_cols and state.merged_df is not None:
        X_clin = _encode_df(state.merged_df[feat_cols])
    else:
        X_clin = pd.DataFrame(index=labels_df.index)

    radiomic_path = RESULTS_DIR / "radiomic_features.csv"
    X_rad = pd.DataFrame()
    if radiomic_path.exists():
        try:
            rad_df = pd.read_csv(radiomic_path, dtype=str)
            rad_df["patient_id"] = rad_df["patient_id"].astype(str).str.strip()
            _RAD_PREFIXES = ("int_", "kin_", "glcm_", "lbp_", "shape_")
            rad_cols = [c for c in rad_df.columns
                        if any(c.startswith(p) for p in _RAD_PREFIXES)]
            rad_sub = rad_df[["patient_id"] + rad_cols].copy()
            labels_with_pid = labels_df.reset_index()
            merged_rad = pd.merge(
                labels_with_pid[["index", "patient_id"]],
                rad_sub, on="patient_id", how="left",
            ).set_index("index")
            merged_rad = merged_rad.drop(columns=["patient_id"], errors="ignore")
            X_rad = _encode_df(merged_rad)
        except Exception as exc:
            log.warning("Radyomik ozellik dosyasi okunamadi: %s", exc)

    frames = [f for f in [X_clin, X_rad] if not f.empty]
    if frames:
        X = pd.concat(frames, axis=1)
        X = X.loc[:, ~X.columns.duplicated()]
    else:
        log.warning("Hic ozellik yok; SHAP atlaniyor.")
        X = pd.DataFrame()

    y      = labels_df["pCR"].astype(int)
    groups = labels_df["patient_id"]

    common_idx = X.index.intersection(y.index)
    X, y, groups = X.loc[common_idx], y.loc[common_idx], groups.loc[common_idx]

    # ── LASSO + MI birleşik özellik seçimi (step_ml_models ile aynı) ────────────
    N_TOP = 25
    top_cols: list[str] = []
    if not X.empty and X.shape[1] > N_TOP:
        scaler   = StandardScaler()
        X_sc     = scaler.fit_transform(X)
        lasso    = LassoCV(cv=min(5, int((y == 1).sum())),
                           max_iter=10000, random_state=RANDOM_SEED, n_jobs=-1)
        lasso.fit(X_sc, y)
        coef_abs = np.abs(lasso.coef_)

        mi_scores = mutual_info_classif(X_sc, y, random_state=RANDOM_SEED, n_neighbors=3)

        def _rank_norm(arr: np.ndarray) -> np.ndarray:
            ranks = arr.argsort().argsort().astype(float)
            return ranks / max(ranks.max(), 1.0)

        combined = _rank_norm(coef_abs) + _rank_norm(mi_scores)
        top_cols = X.columns[np.argsort(combined)[::-1][:N_TOP]].tolist()
        X = X[top_cols]
    elif not X.empty:
        top_cols = X.columns.tolist()

    # ── Stratified train/test split ──────────────────────────────────────────────
    if X.empty or len(y.unique()) < 2:
        log.warning("Yeterli etiketli hasta yok; SHAP atlaniyor.")
    else:
        try:
            X_tr, X_te, y_tr, y_te, pids_tr, pids_te = train_test_split(
                X, y, groups,
                test_size=0.2, random_state=RANDOM_SEED, stratify=y,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te, pids_tr, pids_te = train_test_split(
                X, y, groups, test_size=0.2, random_state=RANDOM_SEED,
            )

        pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
        patient_ids_te = list(pids_te)

        # ── RF ve LightGBM için SHAP ─────────────────────────────────────────────
        SHAP_MODELS = ["random_forest", "lightgbm"]
        for model_name in SHAP_MODELS:
            try:
                pipe = build_pipeline(model_name, pos_weight)
                pipe.fit(X_tr, y_tr)

                # Summary + interaction plots (n_waterfall=0 → sadece özet)
                run_shap_analysis(
                    pipe, X_tr, X_te,
                    model_name=model_name,
                    patient_ids=patient_ids_te,
                    n_waterfall=0,
                )

                # Waterfall için 3 hasta: en iyi pCR, en iyi non-pCR, en belirsiz
                y_prob_te = pipe.predict_proba(X_te)[:, 1]
                te_df = pd.DataFrame({
                    "patient_id": patient_ids_te,
                    "y_true":     y_te.values,
                    "y_prob":     y_prob_te,
                }, index=X_te.index)

                pcr_idx = te_df[te_df.y_true == 1]["y_prob"].idxmax() \
                          if (te_df.y_true == 1).any() else None
                npcr_idx = te_df[te_df.y_true == 0]["y_prob"].idxmin() \
                           if (te_df.y_true == 0).any() else None
                uncertain_idx = (te_df["y_prob"] - 0.5).abs().idxmin()

                for label_tag, idx in [("pCR", pcr_idx),
                                        ("nonpCR", npcr_idx),
                                        ("uncertain", uncertain_idx)]:
                    if idx is None:
                        continue
                    pid = te_df.loc[idx, "patient_id"]
                    try:
                        shap_waterfall(
                            pipe,
                            X_te.loc[[idx]],
                            patient_id=f"{pid}_{label_tag}",
                            model_name=model_name,
                            X_background=X_tr,
                        )
                    except Exception as exc:
                        log.warning("Waterfall [%s/%s] hatasi: %s",
                                    model_name, label_tag, exc)

                outputs[f"shap_{model_name}"] = "ok"
                log.info("SHAP tamamlandi: %s", model_name)
            except Exception as exc:
                log.warning("SHAP [%s] hatasi: %s", model_name, exc)

    # ── GradCAM++ ────────────────────────────────────────────────────────────────
    if labels_df is not None and (MODELS_DIR / "best_cnn.pt").exists():
        try:
            from cnn_model import load_best_model
            device = torch.device("cpu")
            model  = load_best_model(device)
            results = run_gradcam_batch(
                model, labels_df, PROCESSED_DIR, device, max_patients=5,
            )
            if results:
                dice_path = save_dice_report(results)
                outputs["gradcam_dice_report"] = str(dice_path)
                log.info("GradCAM++ tamamlandi: %d hasta", len(results))
        except Exception as exc:
            log.warning("GradCAM hatasi: %s", exc)
    else:
        log.info("best_cnn.pt yok veya labels_df bos; GradCAM atlaniyor.")

    return outputs


# ── Adım 6: visualization ─────────────────────────────────────────────────────

def step_visualization(state: PipelineState) -> dict:
    from visualization import plot_all

    model_preds = state.model_preds or {}
    thresholds  = state.thresholds  or {}
    cv_summary  = state.cv_summary

    # Disk fallback: ml adımı bu oturumda çalışmadıysa kayıtlı verileri yükle
    if not model_preds:
        model_preds, thresholds = _load_model_preds()

    if not model_preds:
        raise ValueError(
            "model_preds bos ve disk cache bulunamadi. "
            "Once '--steps ml' adimini calistirin."
        )

    # cv_summary disk fallback
    if cv_summary is None and (RESULTS_DIR / "ml_results_summary.json").exists():
        cv_summary = json.loads(
            (RESULTS_DIR / "ml_results_summary.json").read_text(encoding="utf-8")
        )
        log.info("cv_summary diskten yuklendi.")

    # merged_df fallback (subgrup analizi için)
    clinical_df = state.merged_df
    if clinical_df is None and (RESULTS_DIR / "merged_catalog.csv").exists():
        try:
            clinical_df = pd.read_csv(RESULTS_DIR / "merged_catalog.csv", dtype=str)
            for col in ("ER", "PR", "HER2", "age", "tumor_size", "grade", "pCR"):
                if col in clinical_df.columns:
                    clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce")
            clinical_df = clinical_df.dropna(subset=["pCR"])
            clinical_df["pCR"] = clinical_df["pCR"].astype(int)
            log.info("clinical_df diskten yuklendi: %d hasta", len(clinical_df))
        except Exception as exc:
            log.warning("merged_catalog.csv okunamadi: %s", exc)

    paths = plot_all(
        model_preds=model_preds,
        cv_summary=cv_summary,
        clinical_df=clinical_df,
        model_probs_sg={n: p for n, (_, p) in model_preds.items()},
        thresholds=thresholds,
    )
    return {k: str(v) for k, v in paths.items() if v}


# ── Demo modu ─────────────────────────────────────────────────────────────────

def _build_demo_state(n: int = 64) -> PipelineState:
    """Gerçek veri olmadan sentetik demo state oluşturur."""
    rng  = np.random.default_rng(RANDOM_SEED)
    pids = [f"ISPY1_{i:03d}" for i in range(n)]

    labels_df = pd.DataFrame({
        "patient_id": pids,
        "pCR":        (rng.random(n) < 0.30).astype(int),
        "ER":         (rng.random(n) > 0.3).astype(int),
        "PR":         (rng.random(n) > 0.4).astype(int),
        "HER2":       (rng.random(n) > 0.75).astype(int),
        "age":        rng.integers(30, 70, n),
        "tumor_size": rng.uniform(1, 8, n),
        "grade":      rng.integers(1, 4, n),
    })

    state = PipelineState()
    state.demo       = True
    state.labels_df  = labels_df[["patient_id", "pCR"]]
    state.merged_df  = labels_df
    state.clinical_df = labels_df
    log.info("Demo modu: n=%d  pCR=%d", n, labels_df["pCR"].sum())
    return state


# ── Özet rapor ────────────────────────────────────────────────────────────────

def _print_summary(step_results: list[StepResult], total_elapsed: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PIPELINE OZETI")
    print(sep)
    for r in step_results:
        icon = {"ok": "[OK]", "failed": "[HATA]",
                "skipped": "[ATLANDI]", "running": "[?]"}.get(r.status, "[ ]")
        err = f"  -> {r.error[:55]}" if r.error else ""
        print(f"  {icon:<10} {r.name:<18} {r.elapsed:>6.1f}s{err}")
    print(sep)
    print(f"  Toplam sure: {total_elapsed:.1f}s")
    n_ok   = sum(1 for r in step_results if r.status == "ok")
    n_fail = sum(1 for r in step_results if r.status == "failed")
    print(f"  Basarili: {n_ok}   Hatali: {n_fail}   Toplam: {len(step_results)}")
    print(sep + "\n")


def _save_summary_report(step_results: list[StepResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "pipeline_summary.json"
    payload = [asdict(r) for r in step_results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("Pipeline raporu: %s", path)
    return path


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

STEP_REGISTRY = {
    "data":    step_data_loader,
    "harmonize": step_harmonize,
    "ml":      step_ml_models,
    "cnn":     step_cnn,
    "shap":    step_shap_gradcam,
    "viz":     step_visualization,
}

ALL_STEPS = ["data", "harmonize", "ml", "cnn", "shap", "viz"]


def run_pipeline(
    steps:      list[str] = ALL_STEPS,
    demo:       bool      = False,
    use_smote:  bool      = True,
) -> list[StepResult]:
    t_start = time.perf_counter()
    state   = _build_demo_state() if demo else PipelineState()
    results: list[StepResult] = []

    for step_name in steps:
        if step_name not in STEP_REGISTRY:
            log.warning("Bilinmeyen adim: %s — atlaniyor.", step_name)
            results.append(StepResult(step_name, status="skipped"))
            continue

        # Demo modunda data/harmonize adımları veri gerektirdiğinden atla
        if demo and step_name in ("data", "harmonize"):
            log.info("[%s] Demo modu — atlaniyor.", step_name)
            results.append(StepResult(step_name, status="skipped"))
            continue

        fn = STEP_REGISTRY[step_name]
        extra = {"use_smote": use_smote} if step_name == "ml" else {}

        step_result, output = _run_step(step_name, fn, state, **extra)

        if output and isinstance(output, dict):
            step_result.outputs = output

        results.append(step_result)

        # Kritik hata → devam et ama logla
        if step_result.status == "failed":
            log.warning("[%s] Adim basarisiz, pipeline devam ediyor.", step_name)

    total = time.perf_counter() - t_start
    _print_summary(results, total)
    _save_summary_report(results)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meme Kanseri NAC Yanit Tahmini Tam Pipeline"
    )
    parser.add_argument(
        "--steps", nargs="+", default=ALL_STEPS,
        choices=list(STEP_REGISTRY),
        help="Calistirilacak adimlar (varsayilan: hepsi)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Gercek veri olmadan sentetik demo calistir",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="ML adiminda SMOTE'u devre disi birak",
    )
    args = parser.parse_args()

    log.info("Pipeline basliyor — adimlar: %s", args.steps)
    log.info("Demo: %s | SMOTE: %s", args.demo, not args.no_smote)

    run_pipeline(
        steps=args.steps,
        demo=args.demo,
        use_smote=not args.no_smote,
    )
