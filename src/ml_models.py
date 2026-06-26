"""
Klasik ML modelleri — 5-fold GroupKFold CV, SMOTE, Youden eşiği, tam metrik seti.

Akış:
  build_model_registry(pos_weight) → model sözlüğü
  run_cv_model(X, y, groups, model_name, ...) → fold sonuçları DataFrame
  run_all_models(X, y, groups, ...)          → 5 model × N fold DataFrame
  save_results(fold_df)                       → CSV + JSON

Komut satırı (test/demo):
  python ml_models.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupKFold, GridSearchCV, LeaveOneOut, StratifiedKFold,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss,
    roc_curve,
)
from sklearn.utils import resample as _bootstrap_resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import BorderlineSMOTE

from config import CV_FOLDS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

LOOCV_THRESHOLD  = 100   # n < bu değerse LOOCV'ye geç
SMOTE_K_MAX      = 5     # SMOTE minimum k_neighbors
MODEL_NAMES      = ["logistic", "svm", "random_forest", "xgboost", "lightgbm", "ensemble",
                    "rf_calibrated", "lgbm_calibrated"]

# İç CV hiperparametre ızgaraları (RF ve LightGBM için)
PARAM_GRIDS: dict[str, dict] = {
    "random_forest": {
        "clf__n_estimators":    [100, 200, 300],
        "clf__max_depth":       [3, 5, 7],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "lightgbm": {
        "clf__n_estimators":   [100, 200],
        "clf__learning_rate":  [0.05, 0.1],
        "clf__num_leaves":     [15, 31],
    },
}

METRIC_COLS = [
    "model", "fold",
    "auc_roc", "auc_pr", "f1",
    "sensitivity", "specificity",
    "mcc", "brier",
    "threshold",
    "n_train", "n_test", "n_pos_train", "n_neg_train",
]


# ── Model kaydı ───────────────────────────────────────────────────────────────

def build_model_registry(pos_weight: float = 1.0) -> dict:
    """
    5 sınıflandırıcıyı sınıf dengesizliğine göre yapılandırır.

    Parameters
    ----------
    pos_weight : n_negative / n_positive  (dengesizlik oranı)
    """
    return {
        "logistic": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            C=1.0,
            random_state=RANDOM_SEED,
        ),
        "svm": SVC(
            probability=True,
            class_weight="balanced",
            kernel="rbf",
            C=1.0,
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            scale_pos_weight=float(pos_weight),
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            class_weight="balanced",
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        # Stacking: RF + LightGBM + XGBoost base; LR meta-learner
        # cross_val_predict (cv=5) ile OOF meta-özellikler üretir
        "ensemble": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
                ("lgbm", LGBMClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    learning_rate=0.05,
                    num_leaves=31,
                    verbose=-1,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
                ("xgb", XGBClassifier(
                    n_estimators=200,
                    scale_pos_weight=float(pos_weight),
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
            ],
            final_estimator=LogisticRegression(
                C=0.1, max_iter=1000, solver="lbfgs",
                random_state=RANDOM_SEED,
            ),
            cv=5,
            passthrough=False,
            n_jobs=1,
        ),
        "rf_calibrated": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            cv=5,
            method="sigmoid",
        ),
        "lgbm_calibrated": CalibratedClassifierCV(
            LGBMClassifier(
                n_estimators=300,
                class_weight="balanced",
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            cv=5,
            method="sigmoid",
        ),
    }


def build_pipeline(model_name: str,
                   pos_weight: float = 1.0) -> Pipeline:
    """StandardScaler + sınıflandırıcı pipeline'ı döner."""
    registry = build_model_registry(pos_weight)
    if model_name not in registry:
        raise ValueError(
            f"Bilinmeyen model: {model_name!r}. "
            f"Secenekler: {list(registry)}"
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    registry[model_name]),
    ])


# ── CV splitter seçimi ────────────────────────────────────────────────────────

def _choose_splitter(n_samples: int,
                     n_splits: int,
                     groups: Optional[np.ndarray]) -> tuple:
    """
    n < LOOCV_THRESHOLD  →  LeaveOneOut
    groups verilmişse    →  GroupKFold
    aksi hâlde           →  StratifiedKFold
    """
    if n_samples < LOOCV_THRESHOLD:
        log.info(
            "n=%d < %d → LOOCV kullaniliyor.",
            n_samples, LOOCV_THRESHOLD,
        )
        return LeaveOneOut(), "loocv"

    if groups is not None:
        n_unique = len(np.unique(groups))
        k = min(n_splits, n_unique)
        log.info("GroupKFold(k=%d) kullaniliyor, %d benzersiz grup.", k, n_unique)
        return GroupKFold(n_splits=k), "group_kfold"

    log.info("StratifiedKFold(k=%d) kullaniliyor.", n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=RANDOM_SEED), "stratified"


# ── SMOTE (yalnızca train fold içinde) ───────────────────────────────────────

def _apply_smote(X_train: np.ndarray,
                 y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    BorderlineSMOTE (borderline-1) ile sentetik örnek üretir.
    Azınlık sınıfı < 2 ise atlanır; k_neighbors azınlık boyutuna göre ayarlanır.
    """
    n_minority = int((y_train == 1).sum())
    if n_minority < 2:
        log.debug("Azinlik sinifi < 2, BorderlineSMOTE atlaniyor.")
        return X_train, y_train

    k = min(SMOTE_K_MAX, n_minority - 1)
    sampler = BorderlineSMOTE(kind="borderline-1", k_neighbors=k, random_state=RANDOM_SEED)
    try:
        return sampler.fit_resample(X_train, y_train)
    except Exception as exc:
        log.warning("BorderlineSMOTE hatasi, orijinal veri kullaniliyor: %s", exc)
        return X_train, y_train


# ── Eşik optimizasyonu ────────────────────────────────────────────────────────

def youden_threshold(y_true: np.ndarray,
                     y_prob: np.ndarray) -> float:
    """ROC eğrisinden Youden J (TPR-FPR) maksimize eden eşiği bulur."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def f1_threshold(y_true: np.ndarray,
                 y_prob: np.ndarray) -> float:
    """
    F1 skorunu maksimize eden eşiği [0.10, 0.90) aralığında arar.
    Dengesiz veri setlerinde Youden-J'ye göre daha yüksek F1 üretir.
    """
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_t = max(
        thresholds,
        key=lambda t: f1_score(y_true, (y_prob >= t).astype(int), zero_division=0),
    )
    return float(best_t)


# ── Bootstrap 95% CI ──────────────────────────────────────────────────────────

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    AUC-ROC için n_bootstrap yeniden-örnekleme ile Bootstrap CI hesaplar.

    Returns
    -------
    {"mean": float, "std": float, "ci_lower": float, "ci_upper": float}
    """
    rng  = np.random.default_rng(random_state)
    idx  = np.arange(len(y_true))
    aucs = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(idx, size=len(idx), replace=True)
        y_b = y_true[boot_idx]
        p_b = y_prob[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_b, p_b)))
        except ValueError:
            pass

    if not aucs:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}

    alpha   = (100 - ci) / 2
    ci_lo   = float(np.percentile(aucs, alpha))
    ci_hi   = float(np.percentile(aucs, 100 - alpha))
    return {
        "mean":     round(float(np.mean(aucs)), 4),
        "std":      round(float(np.std(aucs)),  4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
    }


# ── Metrik hesaplama ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float) -> dict:
    """
    Tek bir (y_true, y_prob) çifti için tam metrik setini hesaplar.

    Returns
    -------
    dict: auc_roc, auc_pr, f1, sensitivity, specificity, mcc, brier, threshold
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix güvenli hesaplama
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auc_pr = float("nan")

    return {
        "auc_roc":     auc_roc,
        "auc_pr":      auc_pr,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc":         float(matthews_corrcoef(y_true, y_pred))
                       if len(np.unique(y_true)) > 1 else float("nan"),
        "brier":       float(brier_score_loss(y_true, y_prob)),
        "threshold":   threshold,
    }


# ── İç CV hiperparametre optimizasyonu ───────────────────────────────────────

def _tuned_pipeline(
    model_name: str,
    pos_weight: float,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> Pipeline:
    """
    RF ve LightGBM için iç 3-fold GridSearchCV ile en iyi hiperparametreleri bulur
    ve klonlanmış (fit edilmemiş) pipeline'ı döner.
    Diğer modeller için doğrudan build_pipeline döner.
    """
    if model_name not in PARAM_GRIDS:
        return build_pipeline(model_name, pos_weight)

    n_pos = int((y_tr == 1).sum())
    if n_pos < 3:
        log.debug("[%s] Azinlik sinifi < 3 — GridSearch atlaniyor.", model_name)
        return build_pipeline(model_name, pos_weight)

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    base = build_pipeline(model_name, pos_weight)

    try:
        gs = GridSearchCV(
            base,
            PARAM_GRIDS[model_name],
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=1,          # alt süreç çakışmasını önlemek için
            refit=True,
            error_score=0.0,
        )
        gs.fit(X_tr, y_tr)
        log.info(
            "  [%s] GridSearch best=%s  inner_AUC=%.3f",
            model_name, gs.best_params_, gs.best_score_,
        )
        return clone(gs.best_estimator_)
    except Exception as exc:
        log.warning("[%s] GridSearch hatasi, varsayilan kullaniliyor: %s", model_name, exc)
        return build_pipeline(model_name, pos_weight)


# ── Tek model CV döngüsü ──────────────────────────────────────────────────────

def run_cv_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    model_name: str = "xgboost",
    use_smote: bool = True,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    """
    Bir model için GroupKFold (veya LOOCV) çapraz doğrulama yapar.

    SMOTE yalnızca train fold içinde uygulanır — test sızıntısı yok.
    Eşik her fold'da bağımsız Youden J ile belirlenir;
    LOOCV modunda eşik tüm OOF tahminlerinden bir kez hesaplanır.

    Parameters
    ----------
    X          : özellik matrisi
    y          : binary hedef (0/1)
    groups     : hasta ID dizisi (GroupKFold için)
    model_name : 'logistic'|'svm'|'random_forest'|'xgboost'|'lightgbm'
    use_smote  : True → train fold'da SMOTE uygula
    n_splits   : GroupKFold için fold sayısı (n<100 ise görmezden gelinir)

    Returns
    -------
    pd.DataFrame — her satır bir fold, sütunlar METRIC_COLS
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    y_arr = y.values if isinstance(y, pd.Series)    else np.array(y)
    g_arr = groups.values if isinstance(groups, pd.Series) and groups is not None \
            else (np.array(groups) if groups is not None else None)

    n_samples   = len(y_arr)
    pos_weight  = float((y_arr == 0).sum()) / max(float((y_arr == 1).sum()), 1)
    splitter, cv_type = _choose_splitter(n_samples, n_splits, g_arr)

    log.info(
        "Model: %-14s  CV: %s  SMOTE: %s  n=%d  pos_weight=%.2f",
        model_name, cv_type, use_smote, n_samples, pos_weight,
    )

    fold_rows: list[dict] = []
    oof_probs  = np.full(n_samples, np.nan)
    oof_true   = np.full(n_samples, np.nan)

    split_kwargs = {"X": X_arr, "y": y_arr}
    if g_arr is not None and cv_type == "group_kfold":
        split_kwargs["groups"] = g_arr

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(**split_kwargs)
    ):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        # Hiperparametre arama: ham (SMOTE öncesi) veri üzerinde, sızıntısız
        pipeline = _tuned_pipeline(model_name, pos_weight, X_tr, y_tr)

        # SMOTE augmentasyonu sonrası nihai fit
        if use_smote:
            X_tr_fit, y_tr_fit = _apply_smote(X_tr, y_tr)
        else:
            X_tr_fit, y_tr_fit = X_tr, y_tr

        pipeline.fit(X_tr_fit, y_tr_fit)

        y_prob_te = pipeline.predict_proba(X_te)[:, 1]

        # OOF birikimi (global eşik ve LOOCV AUC için)
        oof_probs[test_idx] = y_prob_te
        oof_true[test_idx]  = y_te

        n_pos_tr = int((y_tr == 1).sum())
        n_neg_tr = int((y_tr == 0).sum())

        # LOOCV fold'unda tek örnek var → per-fold AUC hesaplanamaz
        if cv_type == "loocv":
            fold_rows.append({
                "model":       model_name,
                "fold":        fold_idx,
                "n_train":     len(y_tr),
                "n_test":      len(y_te),
                "n_pos_train": n_pos_tr,
                "n_neg_train": n_neg_tr,
                # Metrikler LOOCV bittikten sonra doldurulacak
                **{m: np.nan for m in [
                    "auc_roc", "auc_pr", "f1",
                    "sensitivity", "specificity",
                    "mcc", "brier", "threshold",
                ]},
            })
            continue

        # GroupKFold / StratifiedKFold: F1-maximizing threshold per fold
        if len(np.unique(y_te)) < 2:
            thr = 0.5
        else:
            thr = f1_threshold(y_te, y_prob_te)

        metrics = compute_metrics(y_te, y_prob_te, thr)
        fold_rows.append({
            "model": model_name,
            "fold":  fold_idx,
            "n_train":     len(y_tr),
            "n_test":      len(y_te),
            "n_pos_train": n_pos_tr,
            "n_neg_train": n_neg_tr,
            **metrics,
        })

        log.info(
            "  Fold %2d | AUC=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f  Thr=%.3f",
            fold_idx,
            metrics["auc_roc"], metrics["f1"],
            metrics["sensitivity"], metrics["specificity"],
            thr,
        )

    # LOOCV: global eşik ve global metrikler
    if cv_type == "loocv":
        valid = ~np.isnan(oof_probs)
        global_thr = f1_threshold(oof_true[valid], oof_probs[valid])
        global_metrics = compute_metrics(
            oof_true[valid], oof_probs[valid], global_thr
        )
        for row in fold_rows:
            row.update(global_metrics)   # tüm fold'ları global metrikle doldur
        log.info(
            "  LOOCV global | AUC=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f  Thr=%.3f",
            global_metrics["auc_roc"], global_metrics["f1"],
            global_metrics["sensitivity"], global_metrics["specificity"],
            global_thr,
        )

    valid = ~np.isnan(oof_probs)
    oof_dict = {
        "y_true": oof_true[valid].astype(int),
        "y_prob": oof_probs[valid],
    }
    return pd.DataFrame(fold_rows, columns=METRIC_COLS), oof_dict


# ── Tüm modelleri çalıştır ────────────────────────────────────────────────────

def run_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    use_smote: bool = True,
    n_splits: int = CV_FOLDS,
    model_names: list[str] = MODEL_NAMES,
) -> tuple[pd.DataFrame, dict]:
    """
    model_names listesindeki tüm modeller için CV çalıştırır.

    Returns
    -------
    (fold_df, oof_by_model)
    fold_df         : tüm model × fold satırları birleştirilmiş
    oof_by_model    : {model_name: {"y_true": ..., "y_prob": ...}}
    """
    all_folds: list[pd.DataFrame] = []
    oof_by_model: dict = {}

    for name in model_names:
        log.info("=" * 55)
        fold_df, oof_dict = run_cv_model(
            X, y, groups=groups,
            model_name=name,
            use_smote=use_smote,
            n_splits=n_splits,
        )
        all_folds.append(fold_df)
        oof_by_model[name] = oof_dict

    return pd.concat(all_folds, ignore_index=True), oof_by_model


# ── Özet istatistik ───────────────────────────────────────────────────────────

def summarize_results(
    fold_df: pd.DataFrame,
    oof_by_model: Optional[dict] = None,
    n_bootstrap: int = 1000,
) -> dict:
    """
    fold_df'den her model için ortalama ± std hesaplar.
    oof_by_model verilirse AUC için Bootstrap 95% CI de hesaplar.

    Returns
    -------
    {model_name: {metric: {"mean": float, "std": float}, ...}, ...}
    """
    metric_cols = [
        "auc_roc", "auc_pr", "f1",
        "sensitivity", "specificity", "mcc", "brier",
    ]
    summary: dict = {}

    for model_name, grp in fold_df.groupby("model"):
        summary[model_name] = {}
        for col in metric_cols:
            vals = grp[col].dropna()
            summary[model_name][col] = {
                "mean": round(float(vals.mean()), 4),
                "std":  round(float(vals.std()),  4),
            }

        # Bootstrap CI from OOF predictions
        if oof_by_model and model_name in oof_by_model:
            oof = oof_by_model[model_name]
            ci = bootstrap_auc_ci(
                oof["y_true"], oof["y_prob"],
                n_bootstrap=n_bootstrap,
            )
            summary[model_name]["auc_bootstrap"] = ci

    return summary


def print_summary_table(summary: dict) -> None:
    """Özet tablosunu (Bootstrap CI dahil) konsola basar."""
    header = (
        f"{'Model':<15} {'AUC-ROC':>12} {'Bootstrap95CI':>16} "
        f"{'AUC-PR':>8} {'F1':>7} {'Sens':>7} {'Spec':>7} {'MCC':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for model, metrics in summary.items():
        def fmt(key):
            m = metrics[key]["mean"]
            s = metrics[key]["std"]
            return f"{m:.3f}±{s:.3f}"

        ci_str = ""
        if "auc_bootstrap" in metrics:
            ci = metrics["auc_bootstrap"]
            ci_str = f"[{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}]"

        print(
            f"{model:<15} {fmt('auc_roc'):>12} {ci_str:>16} "
            f"{fmt('auc_pr'):>8} {fmt('f1'):>7} {fmt('sensitivity'):>7} "
            f"{fmt('specificity'):>7} {fmt('mcc'):>7}"
        )
    print(sep + "\n")


# ── Model kaydetme / yükleme ──────────────────────────────────────────────────

def save_model(pipeline: Pipeline, name: str) -> Path:
    """Eğitilmiş pipeline'ı MODELS_DIR'e kaydeder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(pipeline, path)
    log.info("Model kaydedildi: %s", path)
    return path


def load_model(name: str) -> Pipeline:
    """Kaydedilmiş pipeline'ı yükler."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {path}")
    return joblib.load(path)


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    use_smote: bool = True,
) -> Pipeline:
    """
    Tüm veri üzerinde (CV olmadan) son modeli eğitir ve kaydeder.
    CV bittikten sonra deployment için kullanılır.
    """
    y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
    pos_weight = float((y_arr == 0).sum()) / max(float((y_arr == 1).sum()), 1)

    X_tr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    y_tr = y_arr

    if use_smote:
        X_tr, y_tr = _apply_smote(X_tr, y_tr)

    pipeline = build_pipeline(model_name, pos_weight)
    pipeline.fit(X_tr, y_tr)
    save_model(pipeline, f"final_{model_name}")
    return pipeline


# ── Sonuç kaydetme ────────────────────────────────────────────────────────────

def save_results(
    fold_df: pd.DataFrame,
    summary: Optional[dict] = None,
    prefix: str = "ml",
) -> tuple[Path, Path]:
    """
    fold_df → CSV, summary → JSON olarak RESULTS_DIR'e kaydeder.

    Returns
    -------
    (csv_path, json_path)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path  = RESULTS_DIR / f"{prefix}_results.csv"
    json_path = RESULTS_DIR / f"{prefix}_results_summary.json"

    fold_df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info("Fold sonuclari kaydedildi: %s", csv_path)

    if summary is None:
        summary = summarize_results(fold_df)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Ozet kaydedildi: %s", json_path)

    return csv_path, json_path


# ── Demo / test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(42)
    n   = 64                                         # ISPY1 gerçekçi hasta sayısı
    n_features = 20

    # Sentetik radyomik + klinik özellikler
    X_demo = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # ~30% pCR oranı (dengesiz)
    y_demo      = pd.Series((rng.random(n) < 0.30).astype(int), name="pCR")
    groups_demo = pd.Series([f"P{i:03d}" for i in range(n)], name="patient_id")

    print(f"Veri: n={n}  pCR={y_demo.sum()}  non-pCR={n - y_demo.sum()}")

    fold_df, oof_by_model = run_all_models(
        X_demo, y_demo,
        groups=groups_demo,
        use_smote=True,
        model_names=MODEL_NAMES,
    )

    summary = summarize_results(fold_df, oof_by_model)
    print_summary_table(summary)

    csv_p, json_p = save_results(fold_df, summary)
    print(f"CSV  : {csv_p}")
    print(f"JSON : {json_p}")
