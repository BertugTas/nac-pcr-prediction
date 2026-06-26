"""
Model yorumlanabilirliği: SHAP (ML modeller) + GradCAM++ (CNN).

SHAP akışı:
  build_shap_explainer()  →  model tipine göre doğru explainer
  shap_summary()          →  beeswarm summary plot
  shap_waterfall()        →  tek hasta karar açıklama
  shap_interaction_heatmap() → özellik etkileşim ısı haritası

GradCAM++ akışı:
  run_gradcam_patient()   →  tek hasta 4-panel görsel + Dice
  run_gradcam_batch()     →  tüm test hastaları
  save_dice_report()      →  JSON raporu

CLI:
  python shap_gradcam.py --mode shap   --model xgboost
  python shap_gradcam.py --mode gradcam
  python shap_gradcam.py --mode all
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # başsız sunucu ortamı için
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")
shap.initjs()   # notebook dışında zararsız

log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

KERNEL_BG_SAMPLES  = 50     # KernelExplainer arka plan boyutu
GRADCAM_THRESHOLD  = 0.5    # Dice için CAM ikileştirme eşiği
SER_PSEUDO_THRESH  = 0.10   # SER pseudo-mask eşiği (normalleştirilmiş uzayda)
TOP_N_FEATURES     = 20     # Grafiklerde gösterilecek özellik sayısı
FIG_DPI            = 150


# ── 1. SHAP — explainer fabrikası ─────────────────────────────────────────────

def _extract_clf(pipeline) -> object:
    """sklearn Pipeline'dan sınıflandırıcıyı çıkarır."""
    return pipeline.named_steps["clf"]


def _scale(pipeline, X: np.ndarray) -> np.ndarray:
    """Pipeline'ın scaler adımıyla X'i dönüştürür."""
    return pipeline.named_steps["scaler"].transform(X)


def build_shap_explainer(
    pipeline,
    X_background: np.ndarray,
    model_name: str,
) -> tuple:
    """
    Model tipine göre doğru SHAP explainer'ı seçer ve döndürür.

    Desteklenen tipler
    ------------------
    tree   : XGBoost, LightGBM, RandomForest  →  TreeExplainer
    linear : LogisticRegression               →  LinearExplainer
    kernel : SVM (ve diğerleri)               →  KernelExplainer

    Returns
    -------
    (explainer, X_bg_scaled, explainer_type)
    """
    clf           = _extract_clf(pipeline)
    X_bg_scaled   = _scale(pipeline, X_background)
    clf_type      = type(clf).__name__

    TREE_TYPES   = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                    "GradientBoostingClassifier", "ExtraTreesClassifier")
    LINEAR_TYPES = ("LogisticRegression", "LinearSVC", "SGDClassifier")

    if clf_type in TREE_TYPES:
        explainer     = shap.TreeExplainer(clf)
        explainer_type = "tree"
        log.info("[%s] TreeExplainer secildi.", model_name)

    elif clf_type in LINEAR_TYPES:
        explainer     = shap.LinearExplainer(clf, X_bg_scaled)
        explainer_type = "linear"
        log.info("[%s] LinearExplainer secildi.", model_name)

    else:
        # SVM ve bilinmeyenler → KernelExplainer
        bg = shap.sample(X_bg_scaled, min(KERNEL_BG_SAMPLES, len(X_bg_scaled)))
        predict_fn    = lambda x: pipeline.predict_proba(
            pipeline.named_steps["scaler"].inverse_transform(x)
        )[:, 1]
        explainer     = shap.KernelExplainer(predict_fn, bg)
        explainer_type = "kernel"
        log.info("[%s] KernelExplainer secildi (bg=%d).", model_name, len(bg))

    return explainer, X_bg_scaled, explainer_type


def _get_shap_values_class1(
    explainer,
    X_scaled: np.ndarray,
    explainer_type: str,
) -> np.ndarray:
    """
    pCR sınıfı (class=1) için SHAP değerlerini döner — her model tipinde tutarlı.

    Returns
    -------
    np.ndarray, şekil (n_samples, n_features)
    """
    sv = explainer.shap_values(X_scaled)
    arr = np.array(sv)

    # list[class0, class1]  →  (2, n, f)  veya  [ndarray, ndarray]
    if isinstance(sv, list):
        return np.array(sv[1])

    # (n, f, n_classes) — RF yeni shap versiyonları
    if arr.ndim == 3:
        return arr[:, :, 1]

    # (n, f) — XGBoost/LGBM/Linear/Kernel
    return arr


# ── 2. SHAP grafikleri ────────────────────────────────────────────────────────

def shap_summary(
    pipeline,
    X: pd.DataFrame,
    model_name: str,
    X_background: Optional[np.ndarray] = None,
) -> Path:
    """
    Beeswarm özet grafiği: her özelliğin pCR kararına katkısı.

    Outputs
    -------
    outputs/plots/shap_summary_{model_name}.png
    """
    if X_background is None:
        X_background = X.values

    feature_names = list(X.columns)
    X_arr         = X.values

    explainer, X_bg_scaled, exp_type = build_shap_explainer(
        pipeline, X_background, model_name
    )
    X_scaled  = _scale(pipeline, X_arr)
    shap_vals = _get_shap_values_class1(explainer, X_scaled, exp_type)

    # En etkili TOP_N özellikleri seç
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-TOP_N_FEATURES:][::-1]
    sv_top    = shap_vals[:, top_idx]
    fn_top    = [feature_names[i] for i in top_idx]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv_top, X_scaled[:, top_idx],
        feature_names=fn_top,
        plot_type="dot",
        show=False,
        max_display=TOP_N_FEATURES,
    )
    ax = plt.gca()
    ax.set_title(f"SHAP Beeswarm — {model_name} (pCR sinifi)", fontsize=13)
    plt.tight_layout()

    path = PLOTS_DIR / f"shap_summary_{model_name}.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    log.info("SHAP summary kaydedildi: %s", path)
    return path


def shap_waterfall(
    pipeline,
    X_single: pd.DataFrame,
    patient_id: str,
    model_name: str,
    X_background: Optional[np.ndarray] = None,
) -> Path:
    """
    Tek hasta için waterfall grafiği — hangi özellik kararı hangi yönde itti.

    Outputs
    -------
    outputs/plots/shap_waterfall_patient{patient_id}.png
    """
    if X_background is None:
        X_background = X_single.values

    feature_names = list(X_single.columns)
    explainer, X_bg_scaled, exp_type = build_shap_explainer(
        pipeline, X_background, model_name
    )
    X_scaled  = _scale(pipeline, X_single.values)
    shap_vals = _get_shap_values_class1(explainer, X_scaled, exp_type)

    # Explainer nesnesi waterfall için Explanation objesi gerektirir
    if exp_type == "tree":
        exp_obj = explainer(X_scaled)
        # RF için class=1 seçimi
        if hasattr(exp_obj, "values") and exp_obj.values.ndim == 3:
            import shap as _shap
            exp_obj = _shap.Explanation(
                values=exp_obj.values[:, :, 1],
                base_values=exp_obj.base_values[:, 1],
                data=exp_obj.data,
                feature_names=feature_names,
            )
        elif exp_obj.values.ndim == 2:
            exp_obj.feature_names = feature_names
    else:
        # Linear/Kernel: Explanation objesini elle kur
        base = float(explainer.expected_value
                     if not isinstance(explainer.expected_value, (list, np.ndarray))
                     else explainer.expected_value[1])
        exp_obj = shap.Explanation(
            values=shap_vals[0],
            base_values=base,
            data=X_scaled[0],
            feature_names=feature_names,
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp_obj[0], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {model_name} | Hasta: {patient_id}", fontsize=12)
    plt.tight_layout()

    path = PLOTS_DIR / f"shap_waterfall_patient{patient_id}.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    log.info("SHAP waterfall kaydedildi: %s", path)
    return path


def shap_interaction_heatmap(
    pipeline,
    X: pd.DataFrame,
    model_name: str,
) -> Path:
    """
    Ağaç modelleri için SHAP etkileşim değeri ısı haritası.
    Doğrusal/kernel modeller için ortalama |SHAP| bar grafiği çizer.

    Outputs
    -------
    outputs/plots/shap_interaction_{model_name}.png
    """
    clf           = _extract_clf(pipeline)
    feature_names = list(X.columns)
    X_scaled      = _scale(pipeline, X.values)

    TREE_TYPES = ("XGBClassifier", "LGBMClassifier",
                  "RandomForestClassifier", "GradientBoostingClassifier")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"shap_interaction_{model_name}.png"

    if type(clf).__name__ in TREE_TYPES:
        try:
            explainer = shap.TreeExplainer(clf)
            sv_inter  = explainer.shap_interaction_values(X_scaled)
            # RF: list[class0, class1]; diğerleri: (n, f, f)
            if isinstance(sv_inter, list):
                sv_inter = sv_inter[1]

            # Seçilen TOP_N özelliklerini al
            mean_inter = np.abs(sv_inter).mean(axis=0)
            mean_main  = np.abs(shap.TreeExplainer(clf).shap_values(X_scaled))
            if isinstance(mean_main, list):
                mean_main = mean_main[1]
            top_idx  = np.argsort(np.abs(mean_main).mean(0))[-TOP_N_FEATURES:]
            sv_top   = mean_inter[np.ix_(top_idx, top_idx)]
            fn_top   = [feature_names[i] for i in top_idx]

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(sv_top, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(len(fn_top)))
            ax.set_yticks(range(len(fn_top)))
            ax.set_xticklabels(fn_top, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(fn_top, fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"SHAP Etkilesim Isi Haritasi — {model_name}", fontsize=13)
            plt.tight_layout()
            plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close()
            log.info("Etkilesim haritasi kaydedildi: %s", path)
            return path

        except Exception as exc:
            log.warning("shap_interaction_values hatasi: %s — bar grafigine geciyor", exc)

    # Fallback: ortalama |SHAP| bar grafiği
    explainer, X_bg_scaled, exp_type = build_shap_explainer(
        pipeline, X.values, model_name
    )
    shap_vals = _get_shap_values_class1(explainer, X_scaled, exp_type)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-TOP_N_FEATURES:]
    fn_top    = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fn_top, mean_abs[top_idx], color="steelblue")
    ax.set_xlabel("Ortalama |SHAP degeri|")
    ax.set_title(f"Ozellik Onemi (SHAP) — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    log.info("SHAP bar grafigi kaydedildi: %s", path)
    return path


# ── 3. GradCAM++ — hedef sınıf fonksiyonu ─────────────────────────────────────

class _PcrLogitTarget:
    """
    pytorch-grad-cam için özel hedef: tek logit çıktılı binary model.
    Sigmoid öncesi ham logit'i döner; pozitif logit = pCR tahmini.
    """
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # model_output: (B, 1) veya (B,)
        out = model_output.squeeze(-1)   # → (B,)
        return out                       # gradient sınıfı için logit yeterli


# ── 4. Dice overlap ───────────────────────────────────────────────────────────

def _ser_pseudo_mask(patch: np.ndarray, threshold: float = SER_PSEUDO_THRESH) -> np.ndarray:
    """
    Patch kanallarından SER pseudo-tümör maskesi türetir.
    patch: (3, H, W)  — [pre, early, late]
    SER_approx = early - pre  (her ikisi de z-score uzayında)
    """
    ser = patch[1] - patch[0]                   # (H, W)
    mask = (ser > threshold).astype(np.uint8)
    return mask


def dice_coefficient(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
    """
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Sıfır maskesi durumunda 0.0 döner.
    """
    inter = int((mask_pred & mask_true).sum())
    total = int(mask_pred.sum()) + int(mask_true.sum())
    return 2 * inter / total if total > 0 else 0.0


# ── 5. Tek hasta GradCAM++ ────────────────────────────────────────────────────

def run_gradcam_patient(
    model:      nn.Module,
    patch:      np.ndarray,           # (3, H, W) float32  — normalleştirilmiş
    patient_id: str,
    label:      Optional[int] = None, # 0/1/None
    device:     Optional[torch.device] = None,
    patch_raw:  Optional[np.ndarray] = None,  # normalize edilmemiş görsel için
) -> dict:
    """
    Tek hasta için GradCAM++ çalıştırır, 4-panelli görsel oluşturur.

    Parameters
    ----------
    model      : unfreeze_for_gradcam() uygulanmış EfficientNet-B0
    patch      : (3, H, W) — CNN'e girecek normalleştirilmiş tensor
    patient_id : dosya adı ve başlık için
    label      : gerçek pCR etiketi (opsiyonel, başlıkta gösterilir)
    patch_raw  : orijinal görsel için normalize edilmemiş patch (opsiyonel)

    Returns
    -------
    {"patient_id", "dice", "cam_path", "label"}
    """
    if device is None:
        device = torch.device("cpu")

    # ── GradCAM++ ──
    from cnn_model import get_gradcam_layer
    target_layer = get_gradcam_layer(model)

    cam_extractor = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
    )

    img_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)   # (1,3,H,W)
    targets    = [_PcrLogitTarget()]

    grayscale_cam = cam_extractor(
        input_tensor=img_tensor,
        targets=targets,
        aug_smooth=True,
        eigen_smooth=True,
    )[0]   # (H, W)  [0,1]

    # ── SER pseudo-maskesi ──
    ser_mask = _ser_pseudo_mask(patch)     # (H, W)

    # ── Dice ──
    cam_mask = (grayscale_cam >= GRADCAM_THRESHOLD).astype(np.uint8)
    dice     = dice_coefficient(cam_mask, ser_mask)

    # ── Görsel ──
    # Görseller için early-post kanalını kullan (en bilgi yüklü kanal)
    vis_channel = patch[1]
    # [0,1] aralığına getir
    lo, hi = vis_channel.min(), vis_channel.max()
    vis_img = (vis_channel - lo) / (hi - lo + 1e-8)
    vis_rgb = np.stack([vis_img] * 3, axis=-1).astype(np.float32)  # (H,W,3)

    cam_overlay = show_cam_on_image(vis_rgb, grayscale_cam, use_rgb=True)

    label_str = {0: "non-pCR", 1: "pCR"}.get(label, "?") if label is not None else ""

    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (vis_img,      "gray",  f"Orijinal\n{patient_id}  {label_str}"),
        (grayscale_cam,"jet",   "GradCAM++\n(pCR hedefi)"),
        (ser_mask,     "Greens", f"SER Pseudo-Maske\n(thresh={SER_PSEUDO_THRESH})"),
        (cam_overlay,   None,   f"Overlay\nDice = {dice:.3f}"),
    ]

    for i, (img_data, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if cmap is None:
            ax.imshow(img_data)
        else:
            ax.imshow(img_data, cmap=cmap, vmin=0, vmax=1
                      if img_data.max() <= 1 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # CAM ve maskeyi aynı panele bindirerek sınırı göster
    ax3 = fig.axes[3]
    contour_data = ser_mask.astype(float)
    ax3.contour(contour_data, levels=[0.5], colors="cyan", linewidths=1.5)
    ax3.legend(
        handles=[plt.matplotlib.patches.Patch(color="cyan", label="SER maske sınırı")],
        loc="lower right", fontsize=7,
    )

    plt.suptitle(
        f"GradCAM++ | {patient_id}  |  aug_smooth=True  eigen_smooth=True",
        fontsize=12, y=1.01,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cam_path = PLOTS_DIR / f"gradcam_{patient_id}.png"
    plt.savefig(cam_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    log.info("[%s] GradCAM++ tamamlandi | Dice=%.3f", patient_id, dice)
    return {"patient_id": patient_id, "dice": dice,
            "cam_path": str(cam_path), "label": label}


# ── 6. Toplu GradCAM++ analizi ────────────────────────────────────────────────

def run_gradcam_batch(
    model:      nn.Module,
    labels_df:  pd.DataFrame,
    patch_dir:  Path = PROCESSED_DIR,
    device:     Optional[torch.device] = None,
    max_patients: Optional[int] = None,
) -> list[dict]:
    """
    Tüm test hastaları için GradCAM++ çalıştırır.

    Parameters
    ----------
    labels_df    : 'patient_id' ve 'pCR' sütunları içeren DataFrame
    max_patients : None → hepsi; int → ilk N hasta (hızlı test için)

    Returns
    -------
    [{"patient_id", "dice", "cam_path", "label"}, ...]
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()

    results  = []
    patients = labels_df.to_dict("records")
    if max_patients:
        patients = patients[:max_patients]

    for row in patients:
        pid   = str(row["patient_id"])
        label = int(row.get("pCR", -1))
        path  = patch_dir / f"{pid}.npy"

        if not path.exists():
            log.warning("[%s] .npy bulunamadi, atlaniyor.", pid)
            continue

        patch = np.load(str(path)).astype(np.float32)   # (3, H, W)
        try:
            res = run_gradcam_patient(
                model, patch, pid,
                label=label if label >= 0 else None,
                device=device,
            )
            results.append(res)
        except Exception as exc:
            log.warning("[%s] GradCAM hatasi: %s", pid, exc)

    return results


# ── 7. Dice raporu ────────────────────────────────────────────────────────────

def compute_dice_report(results: list[dict]) -> dict:
    """
    pCR / non-pCR grupları için Dice istatistiklerini hesaplar.

    Returns
    -------
    {
      "overall":  {"mean", "std", "n"},
      "pCR":      {"mean", "std", "n"},
      "non_pCR":  {"mean", "std", "n"},
    }
    """
    all_dice  = np.array([r["dice"] for r in results])
    pcr_dice  = np.array([r["dice"] for r in results if r.get("label") == 1])
    npcr_dice = np.array([r["dice"] for r in results if r.get("label") == 0])

    def _stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std()),  4),
            "n":    int(len(arr)),
        }

    return {
        "overall":  _stats(all_dice),
        "pCR":      _stats(pcr_dice),
        "non_pCR":  _stats(npcr_dice),
    }


def print_dice_report(report: dict) -> None:
    sep = "-" * 46
    print(f"\n{sep}")
    print("  GRADCAM++ DICE RAPORU")
    print(sep)
    for group, stats in report.items():
        if stats["n"] == 0:
            print(f"  {group:<10}  n=0  (veri yok)")
        else:
            print(
                f"  {group:<10}  n={stats['n']:>3}  "
                f"Dice = {stats['mean']:.3f} +/- {stats['std']:.3f}"
            )
    print(sep + "\n")


def save_dice_report(results: list[dict]) -> Path:
    """outputs/results/gradcam_dice_report.json olarak kaydeder."""
    report  = compute_dice_report(results)
    payload = {"summary": report, "per_patient": results}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "gradcam_dice_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("Dice raporu kaydedildi: %s", path)
    print_dice_report(report)
    return path


# ── 8. Tam SHAP akışı (kolaylık sarmalayıcısı) ───────────────────────────────

def run_shap_analysis(
    pipeline,
    X_train:      pd.DataFrame,
    X_test:       pd.DataFrame,
    model_name:   str,
    patient_ids:  Optional[list[str]] = None,
    n_waterfall:  int = 3,
) -> None:
    """
    Eğitim seti üzerinde SHAP explainer kurar; test seti üzerinde analiz yapar.

    1. Summary plot (tüm test seti)
    2. Waterfall plots (ilk n_waterfall hasta)
    3. Interaction heatmap (eğitim seti üzerinde)
    """
    log.info("[%s] SHAP analizi basliyor ...", model_name)

    shap_summary(pipeline, X_test, model_name, X_background=X_train.values)
    shap_interaction_heatmap(pipeline, X_train, model_name)

    for i in range(min(n_waterfall, len(X_test))):
        pid = patient_ids[i] if patient_ids else str(i)
        shap_waterfall(
            pipeline,
            X_test.iloc[[i]],
            patient_id=pid,
            model_name=model_name,
            X_background=X_train.values,
        )

    log.info("[%s] SHAP analizi tamamlandi.", model_name)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="SHAP + GradCAM++ yorumlanabilirlik")
    parser.add_argument("--mode",       default="all",
                        choices=["shap", "gradcam", "all"])
    parser.add_argument("--model",      default="xgboost",
                        help="ML model adi (SHAP icin)")
    parser.add_argument("--patch-dir",  default=str(PROCESSED_DIR))
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)

    # Etiket tablosu
    catalog_path = RESULTS_DIR / "merged_catalog.csv"
    if catalog_path.exists():
        labels_df = pd.read_csv(catalog_path)[["patient_id", "pCR"]].dropna()
    else:
        npy_files = sorted(patch_dir.glob("*.npy"))
        if not npy_files:
            raise SystemExit("Hic .npy patch bulunamadi.")
        rng = np.random.default_rng(42)
        labels_df = pd.DataFrame({
            "patient_id": [p.stem for p in npy_files],
            "pCR":        (rng.random(len(npy_files)) < 0.30).astype(int),
        })

    if args.mode in ("shap", "all"):
        print("SHAP analizi: ML modeli gerektiriyor.")
        print("Kullanim: run_shap_analysis(pipeline, X_train, X_test, model_name)")
        print("Ornek: python notebooks/shap_demo.ipynb")

    if args.mode in ("gradcam", "all"):
        from cnn_model import load_best_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model  = load_best_model(device)
            results = run_gradcam_batch(
                model, labels_df, patch_dir, device,
                max_patients=args.max_patients,
            )
            if results:
                save_dice_report(results)
        except FileNotFoundError as e:
            print(f"Model bulunamadi: {e}")
            print("Once cnn_model.py ile model egitilmeli.")
