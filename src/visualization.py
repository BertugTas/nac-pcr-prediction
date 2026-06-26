"""
Tüm proje grafikleri — her fonksiyon PNG dosyası kaydeder.

Temel API:
  plot_roc_all_models(model_preds)         →  roc_all_models.png
  plot_pr_all_models(model_preds)          →  pr_all_models.png
  plot_confusion_matrices(model_preds)     →  cm_{model}.png (her model)
  plot_model_comparison_table(summary)     →  model_comparison.png
  plot_calibration(model_preds)            →  calibration.png
  plot_subgroup_analysis(df, model_preds)  →  subgroup_auc.png
  plot_all(...)                            →  tümünü çağırır

Her fonksiyon kaydedilen Path'i döner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

from config import PLOTS_DIR

log = logging.getLogger(__name__)

# ── Stil ──────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.05)

MODEL_COLORS = {
    "logistic":      "#4C72B0",
    "svm":           "#DD8452",
    "random_forest": "#55A868",
    "xgboost":       "#C44E52",
    "lightgbm":      "#8172B3",
    "cnn":           "#937860",
}
DEFAULT_COLOR = "#666666"
FIG_DPI = 150

# ModelPreds tip takma adı: {model_name: (y_true, y_prob)}
ModelPreds = dict[str, tuple[np.ndarray, np.ndarray]]


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Kaydedildi: %s", path)
    return path


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, DEFAULT_COLOR)


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


# ── 1. ROC eğrisi — tüm modeller tek grafikte ─────────────────────────────────

def plot_roc_all_models(
    model_preds: ModelPreds,
    title: str = "ROC Egrileri — Tum Modeller",
) -> Path:
    """
    Tüm modellerin ROC eğrisini tek eksen üzerinde çizer.
    Her eğrinin yanında AUC değeri gösterilir.

    Parameters
    ----------
    model_preds : {model_name: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (y_true, y_prob) in sorted(model_preds.items()):
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                color=_color(name),
                label=f"{name}  (AUC = {roc_auc:.3f})")

    # Şans çizgisi
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Sans (0.500)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Yanlış Pozitif Oranı (1 - Özgüllük)", fontsize=11)
    ax.set_ylabel("Doğru Pozitif Oranı (Duyarlılık)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return _savefig(fig, "roc_all_models.png")


# ── 2. Precision-Recall eğrisi — tüm modeller ────────────────────────────────

def plot_pr_all_models(
    model_preds: ModelPreds,
    title: str = "Precision-Recall Egrileri",
) -> Path:
    """
    Her modelin PR eğrisini ve AP değerini tek grafik üzerinde gösterir.
    Noktalı yatay çizgi: sınıf frekansı (rastgele sınıflandırıcı taban çizgisi).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    baseline = None
    for name, (y_true, y_prob) in sorted(model_preds.items()):
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2,
                color=_color(name),
                label=f"{name}  (AP = {ap:.3f})")
        if baseline is None:
            baseline = float(y_true.mean())

    if baseline is not None:
        ax.axhline(baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                   label=f"Taban ({baseline:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall (Duyarlılık)", fontsize=11)
    ax.set_ylabel("Precision (Kesinlik)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return _savefig(fig, "pr_all_models.png")


# ── 3. Konfüzyon matrisleri (normalize edilmiş) ───────────────────────────────

def plot_confusion_matrices(
    model_preds: ModelPreds,
    thresholds: Optional[dict[str, float]] = None,
) -> list[Path]:
    """
    Her model için normalize edilmiş konfüzyon matrisi kaydeder.
    Threshold verilmezse Youden J ile belirlenir.

    Returns
    -------
    Kaydedilen dosya yollarının listesi
    """
    paths = []
    n_models = len(model_preds)
    if n_models == 0:
        return paths

    # Tüm matrisleri tek canvas'a yan yana çiz
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for ax_idx, (name, (y_true, y_prob)) in enumerate(sorted(model_preds.items())):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        thr = (thresholds or {}).get(name) or (
            _youden_threshold(y_true, y_prob)
            if len(np.unique(y_true)) > 1 else 0.5
        )
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["non-pCR", "pCR"],
            yticklabels=["non-pCR", "pCR"],
            vmin=0, vmax=1, linewidths=0.5,
            ax=ax, cbar=False,
        )
        # Ham sayıları ek bilgi olarak göster
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75,
                        f"n={cm[i,j]}",
                        ha="center", va="center",
                        fontsize=8, color="gray")
        ax.set_xlabel("Tahmin", fontsize=10)
        ax.set_ylabel("Gerçek", fontsize=10)
        ax.set_title(f"{name}\n(thr={thr:.2f})", fontsize=11, fontweight="bold")

    # Boş eksenler
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.suptitle("Konfüzyon Matrisleri (Normalize)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = _savefig(fig, "confusion_matrices.png")
    paths.append(path)
    return paths


# ── 4. Model karşılaştırma tablosu ────────────────────────────────────────────

def plot_model_comparison_table(
    summary: dict,
    metrics: list[str] | None = None,
) -> Path:
    """
    CV özet tablosunu renk kodlu ısı haritası olarak çizer.

    Parameters
    ----------
    summary : {model_name: {metric: {"mean": float, "std": float}}}
              ml_models.summarize_results() çıktısı
    metrics : gösterilecek metrikler (None → varsayılan set)
    """
    if metrics is None:
        metrics = ["auc_roc", "auc_pr", "f1", "sensitivity", "specificity", "mcc"]

    models = sorted(summary.keys())
    data   = np.zeros((len(models), len(metrics)))
    annot  = np.empty_like(data, dtype=object)

    for r, model in enumerate(models):
        for c, metric in enumerate(metrics):
            stats = summary[model].get(metric, {})
            mean_ = stats.get("mean", float("nan"))
            std_  = stats.get("std",  float("nan"))
            data[r, c]  = mean_ if not np.isnan(mean_) else 0
            annot[r, c] = (f"{mean_:.3f}\n±{std_:.3f}"
                           if not np.isnan(mean_) else "N/A")

    metric_labels = {
        "auc_roc": "AUC-ROC", "auc_pr": "AUC-PR",
        "f1": "F1", "sensitivity": "Sens.",
        "specificity": "Spec.", "mcc": "MCC", "brier": "Brier",
    }
    col_labels = [metric_labels.get(m, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(2.2 * len(metrics), 0.9 * len(models) + 2))
    sns.heatmap(
        data, annot=annot, fmt="",
        xticklabels=col_labels, yticklabels=models,
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9}, ax=ax,
    )
    ax.set_title("Model Karsilastirma (CV Ortalama ± Std)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Metrik", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _savefig(fig, "model_comparison.png")


# ── 5. Kalibrasyon eğrisi + reliability diagram ───────────────────────────────

def plot_calibration(
    model_preds: ModelPreds,
    n_bins: int = 8,
) -> Path:
    """
    Her modelin kalibrasyon eğrisini ve Brier skorunu gösterir.
    Mükemmel kalibrasyon: y = x köşegeni.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Mükemmel kalibrasyon")
    brier_scores: dict[str, float] = {}

    for name, (y_true, y_prob) in sorted(model_preds.items()):
        if len(np.unique(y_true)) < 2:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="uniform"
            )
        except ValueError:
            continue
        bs = brier_score_loss(y_true, y_prob)
        brier_scores[name] = bs
        ax1.plot(mean_pred, frac_pos, "o-", lw=2,
                 color=_color(name),
                 label=f"{name}  (Brier={bs:.3f})", markersize=5)

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.08)
    ax1.set_xlabel("Ortalama Tahmin Olasılığı", fontsize=11)
    ax1.set_ylabel("Gerçek Pozitif Fraksiyonu", fontsize=11)
    ax1.set_title("Kalibrasyon Eğrisi (Reliability Diagram)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # Sağ: Brier skoru karşılaştırması (düşük = iyi)
    if brier_scores:
        names_sorted = sorted(brier_scores, key=brier_scores.get)
        vals_sorted  = [brier_scores[n] for n in names_sorted]
        colors_bar   = [_color(n) for n in names_sorted]
        bars = ax2.barh(names_sorted, vals_sorted, color=colors_bar, edgecolor="white")
        ax2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax2.axvline(0.25, color="gray", linestyle=":", lw=1.5, alpha=0.7,
                    label="0.25 referans")
        ax2.set_xlabel("Brier Skoru (dusuk = iyi)", fontsize=11)
        ax2.set_title("Brier Skoru Karsilastirmasi", fontsize=12, fontweight="bold")
        ax2.set_xlim(0, max(vals_sorted) * 1.25 + 0.05)
        ax2.legend(fontsize=9)

    fig.suptitle("Model Kalibrasyonu", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _savefig(fig, "calibration.png")


# ── 6. Subgrup analizi ────────────────────────────────────────────────────────

def _assign_subgroup(row: pd.Series) -> str:
    """ER, PR, HER2 sütunlarından moleküler alt tip etiketler."""
    def _pos(col):
        if col not in row or pd.isna(row[col]):
            return None
        return str(row[col]).strip().lower() in ("1", "positive", "pos", "yes", "+", "true")

    er  = _pos("ER")
    pr  = _pos("PR")
    her2 = _pos("HER2")

    if her2 is True:
        return "HER2+"
    if er is False and pr is False and (her2 is False or her2 is None):
        return "Triple-Negatif"
    if er is True or pr is True:
        return "Luminal"
    return "Bilinmeyen"


def plot_subgroup_analysis(
    clinical_df: pd.DataFrame,
    model_probs: dict[str, np.ndarray],
    title: str = "Subgrup AUC Analizi",
) -> Path:
    """
    HER2+, Triple-negatif ve Luminal alt gruplarında model AUC'larını karşılaştırır.

    Parameters
    ----------
    clinical_df  : 'pCR', 'ER', 'PR', 'HER2' sütunları içeren DataFrame
                   (satır sırası model_probs dizileriyle eşleşmeli)
    model_probs  : {model_name: y_prob array} — uzunluk = len(clinical_df)
    """
    df = clinical_df.copy().reset_index(drop=True)
    df["subgroup"] = df.apply(_assign_subgroup, axis=1)

    subgroups = [s for s in ["HER2+", "Triple-Negatif", "Luminal"]
                 if (df["subgroup"] == s).sum() >= 5]

    if not subgroups:
        log.warning("Yeterli subgrup verisi yok (min 5 hasta). "
                    "ER/PR/HER2 sutunlarini kontrol edin.")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Yeterli subgrup verisi yok",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return _savefig(fig, "subgroup_auc.png")

    models  = sorted(model_probs.keys())
    n_sub   = len(subgroups)
    n_mod   = len(models)
    x       = np.arange(n_sub)
    bar_w   = 0.8 / n_mod
    offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod) * bar_w

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_sub), 5.5))

    for m_idx, model_name in enumerate(models):
        y_prob = np.array(model_probs[model_name])
        aucs, errs, ns = [], [], []

        for sg in subgroups:
            mask  = (df["subgroup"] == sg).values
            n_sg  = int(mask.sum())
            y_t   = df.loc[mask, "pCR"].astype(int).values
            y_p   = y_prob[mask]

            if len(np.unique(y_t)) < 2 or n_sg < 5:
                aucs.append(0.0); errs.append(0.0); ns.append(n_sg)
                continue

            a = float(roc_auc_score(y_t, y_p))
            # Bootstrapped 95% CI için ±std tahmini
            boot = [
                roc_auc_score(
                    y_t[idx := np.random.choice(n_sg, n_sg, replace=True)],
                    y_p[idx],
                )
                for _ in range(200)
                if len(np.unique(y_t[np.random.choice(n_sg, n_sg, replace=True)])) > 1
            ]
            aucs.append(a)
            errs.append(np.std(boot) if boot else 0.0)
            ns.append(n_sg)

        bars = ax.bar(
            x + offsets[m_idx], aucs,
            width=bar_w * 0.9,
            color=_color(model_name),
            label=model_name,
            yerr=errs,
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )
        for b, n_sg in zip(bars, ns):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(errs) + 0.02,
                    f"n={n_sg}", ha="center", va="bottom",
                    fontsize=7, color="gray")

    ax.axhline(0.5, color="k", linestyle="--", lw=1.2, alpha=0.5, label="Sans (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85, ncol=2)
    fig.tight_layout()
    return _savefig(fig, "subgroup_auc.png")


# ── 7. Tüm grafikleri üret ────────────────────────────────────────────────────

def plot_all(
    model_preds:    ModelPreds,
    cv_summary:     Optional[dict]        = None,
    clinical_df:    Optional[pd.DataFrame] = None,
    model_probs_sg: Optional[dict[str, np.ndarray]] = None,
    thresholds:     Optional[dict[str, float]] = None,
) -> dict[str, Path]:
    """
    Tüm grafikleri üretir ve kaydedilen yolları döner.

    Parameters
    ----------
    model_preds    : {model_name: (y_true, y_prob)}
    cv_summary     : ml_models.summarize_results() çıktısı (karşılaştırma tablosu)
    clinical_df    : subgrup analizi için klinik DataFrame
    model_probs_sg : subgrup analizi için {model: y_prob} (satırlar clinical_df ile eşleşmeli)
    thresholds     : {model_name: float} — konfüzyon matrisi için eşikler

    Returns
    -------
    {"grafik_adi": Path, ...}
    """
    paths: dict[str, Path] = {}

    def _run(key, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, list):
                paths[key] = result[0] if result else None
            else:
                paths[key] = result
            log.info("Grafik olusturuldu: %s", key)
        except Exception as exc:
            log.warning("Grafik HATASI [%s]: %s", key, exc)

    _run("roc",         plot_roc_all_models,       model_preds)
    _run("pr",          plot_pr_all_models,         model_preds)
    _run("cm",          plot_confusion_matrices,    model_preds, thresholds)
    _run("calibration", plot_calibration,           model_preds)

    if cv_summary:
        _run("comparison", plot_model_comparison_table, cv_summary)

    if clinical_df is not None and model_probs_sg:
        _run("subgroup", plot_subgroup_analysis, clinical_df, model_probs_sg)

    log.info("Tum grafikler tamamlandi — toplam %d dosya.", len(paths))
    return paths
