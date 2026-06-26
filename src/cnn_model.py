"""
EfficientNet-B0 transfer learning — DCE-MRI patch ile NAC yanıt tahmini.

Akış:
  PatchDataset          →  .npy yükle + augmentasyon
  build_efficientnet_b0 →  dondurulmuş backbone + özel FC baş
  train_fold            →  BCEWithLogitsLoss + Adam + erken durdurma
  run_cv_training       →  5-fold GroupKFold CV
  get_gradcam_layer     →  GradCAM için hedef katman

CLI:
  python cnn_model.py --fold 0       (tek fold)
  python cnn_model.py                (tüm fold'lar)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import torchvision.transforms.functional as TF
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    matthews_corrcoef, roc_curve,
)

from config import (
    CV_FOLDS, MODELS_DIR, PROCESSED_DIR,
    RANDOM_SEED, RESULTS_DIR,
)

log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

PATCH_SIZE    = 128
IN_CHANNELS   = 3       # pre / early / late DCE fazları
DROPOUT_HEAD  = 0.5
LR_DEFAULT    = 1e-3
BATCH_SIZE    = 16
MAX_EPOCHS    = 100
PATIENCE      = 10
NOISE_STD     = 0.02    # Gaussian noise std (normalleştirilmiş uzayda)
ROT_DEGREES   = 10

# ImageNet istatistikleri (EfficientNet için)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BEST_MODEL_PATH = MODELS_DIR / "best_cnn.pt"


# ── 1. Augmentasyon ───────────────────────────────────────────────────────────

class GaussianNoise(nn.Module):
    """Eğitim sırasında görüntüye Gaussian gürültü ekler."""
    def __init__(self, std: float = NOISE_STD):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


def _normalize_tensor(img: torch.Tensor) -> torch.Tensor:
    """ImageNet istatistikleriyle kanal bazlı normalize eder."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=img.dtype).view(3, 1, 1)
    return (img - mean) / std


class MedicalAugment:
    """
    Medikal görüntüye uygun augmentasyon.

    Yapılanlar   : yatay flip, ±10° rotasyon, Gaussian gürültü
    Yapılmayanlar: dikey flip (anatomik yön bozulur), renk jitter
                   (yoğunluk harmonize.py'de kalibre edildi)
    """
    def __init__(self, p_flip: float = 0.5,
                 degrees: float = ROT_DEGREES,
                 noise_std: float = NOISE_STD):
        self.p_flip    = p_flip
        self.degrees   = degrees
        self.noise_std = noise_std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Yatay flip
        if random.random() < self.p_flip:
            img = TF.hflip(img)

        # Rotasyon
        angle = random.uniform(-self.degrees, self.degrees)
        img = TF.rotate(img, angle)

        # Gaussian gürültü
        img = img + torch.randn_like(img) * self.noise_std

        return img


def get_transforms(train: bool) -> callable:
    """
    Eğitim: augmentasyon + ImageNet normalizasyon
    Doğrulama: yalnızca ImageNet normalizasyon
    """
    if train:
        augment = MedicalAugment()
        def transform(img: torch.Tensor) -> torch.Tensor:
            img = augment(img)
            return _normalize_tensor(img)
        return transform

    def transform(img: torch.Tensor) -> torch.Tensor:
        return _normalize_tensor(img)
    return transform


# ── 2. Dataset ────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    data/processed/ altındaki .npy patch dosyalarını yükler.

    Her .npy dosyası: (3, 128, 128) float32
      Kanal 0: pre-contrast
      Kanal 1: early post-contrast
      Kanal 2: late post-contrast

    Parameters
    ----------
    patch_dir  : .npy dosyalarının bulunduğu klasör
    labels_df  : 'patient_id' ve 'pCR' sütunları içeren DataFrame
    train      : True → augmentasyon uygulanır
    """

    def __init__(
        self,
        patch_dir: Path,
        labels_df: pd.DataFrame,
        train: bool = True,
    ):
        self.transform = get_transforms(train)
        self.samples: list[tuple[Path, int]] = []   # (npy_path, label)

        for _, row in labels_df.iterrows():
            pid   = str(row["patient_id"])
            label = int(row["pCR"])
            path  = patch_dir / f"{pid}.npy"
            if path.exists():
                self.samples.append((path, label))
            else:
                log.warning("Patch bulunamadi, atlaniyor: %s", path)

        if not self.samples:
            raise FileNotFoundError(
                f"Hic .npy patch bulunamadi: {patch_dir}\n"
                "Oncelikle harmonize.py calistirin."
            )

        log.info(
            "%s seti: %d ornek  (pCR=%d  non-pCR=%d)",
            "Train" if train else "Val",
            len(self.samples),
            sum(l for _, l in self.samples),
            sum(1 - l for _, l in self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        patch = np.load(str(path))                          # (3, H, W) float32
        img   = torch.from_numpy(patch)                     # float32 tensor
        img   = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    def get_patient_ids(self) -> list[str]:
        return [p.stem for p, _ in self.samples]

    def get_labels(self) -> np.ndarray:
        return np.array([l for _, l in self.samples])


# ── 3. GroupKFold split yardımcıları ─────────────────────────────────────────

def create_fold_splits(
    labels_df: pd.DataFrame,
    patch_dir: Path = PROCESSED_DIR,
    n_splits: int = CV_FOLDS,
) -> list[tuple[list[int], list[int]]]:
    """
    Var olan .npy patch'lere sahip hastaları GroupKFold ile böler.

    Returns
    -------
    [(train_indices, val_indices), ...]  — uzunluk n_splits
    """
    available = [
        row for _, row in labels_df.iterrows()
        if (patch_dir / f"{row['patient_id']}.npy").exists()
    ]
    if not available:
        raise FileNotFoundError("Hic islenmis patch bulunamadi.")

    df_avail = pd.DataFrame(available).reset_index(drop=True)
    X_dummy  = np.zeros(len(df_avail))
    y_arr    = df_avail["pCR"].astype(int).values
    groups   = df_avail["patient_id"].values

    gkf    = GroupKFold(n_splits=min(n_splits, len(df_avail)))
    splits = list(gkf.split(X_dummy, y_arr, groups))
    return splits, df_avail


# ── 4. Model ──────────────────────────────────────────────────────────────────

def build_efficientnet_b0(freeze_backbone: bool = True) -> nn.Module:
    """
    EfficientNet-B0 transfer learning.

    Mimarı:
      features[0-8] : ImageNet ağırlıklı konvolüsyon omurgası (dondurulmuş)
      avgpool        : adaptif ortalama havuzlama  (1×1)
      classifier     : Dropout(0.5) → Linear(1280,256) → SiLU →
                       Dropout(0.5) → Linear(256,1)    [logit çıktı]

    Çıktı logit'tir; BCEWithLogitsLoss ile kullanılır.
    Olasılık için: torch.sigmoid(logit)
    """
    backbone = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    if freeze_backbone:
        for param in backbone.features.parameters():
            param.requires_grad = False

    in_features = backbone.classifier[1].in_features   # 1280

    backbone.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_HEAD),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=DROPOUT_HEAD),
        nn.Linear(256, 1),                 # tek logit çıktı
    )

    return backbone


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    """
    GradCAM için hedef katmanı döner.
    EfficientNet-B0'da son konvolüsyon bloğu: features[8]
    (1280 kanallı 4×4 feature map üretir 128×128 giriş için)
    """
    return model.features[8]


def unfreeze_for_gradcam(model: nn.Module) -> nn.Module:
    """
    Inference sırasında son konvolüsyon bloğunun gradyanlarını açar;
    GradCAM'in backward pass yapabilmesi için gereklidir.
    """
    for param in model.features[8].parameters():
        param.requires_grad = True
    return model


# ── 5. Kayıp fonksiyonu ───────────────────────────────────────────────────────

def build_criterion(y_train: np.ndarray, device: torch.device) -> nn.Module:
    """
    BCEWithLogitsLoss + pos_weight ile sınıf dengesizliğini dengeler.
    pos_weight = n_negative / n_positive
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    log.info("pos_weight=%.2f  (n_pos=%d  n_neg=%d)", pos_w.item(), n_pos, n_neg)
    return nn.BCEWithLogitsLoss(pos_weight=pos_w)


# ── 6. Erken durdurma ─────────────────────────────────────────────────────────

@dataclass
class EarlyStopping:
    patience:  int   = PATIENCE
    min_delta: float = 1e-4
    _counter:  int   = field(default=0, init=False, repr=False)
    _best:     float = field(default=-np.inf, init=False, repr=False)
    stopped:   bool  = field(default=False, init=False)

    def step(self, metric: float) -> bool:
        """
        True döndürürse eğitimi durdur.
        metric büyüdükçe iyi (AUC için uygundur).
        """
        if metric > self._best + self.min_delta:
            self._best   = metric
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.stopped = True
        return self.stopped

    @property
    def best(self) -> float:
        return self._best


# ── 7. Tek epoch eğitim / doğrulama ──────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)          # (B,)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    (val_loss, y_true, y_prob)
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(1)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    y_prob = np.array(all_probs, dtype=np.float32)
    y_true = np.array(all_labels, dtype=np.int32)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, y_true, y_prob


# ── 8. Metrik hesaplama ───────────────────────────────────────────────────────

def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def compute_val_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUC-ROC, F1, Sensitivity, Specificity, MCC."""
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in
                ["auc_roc", "f1", "sensitivity", "specificity", "mcc", "threshold"]}

    auc = float(roc_auc_score(y_true, y_prob))
    thr = _youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc_roc":     round(auc, 4),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "mcc":         round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "threshold":   round(float(thr), 4),
    }


# ── 9. Tek fold eğitimi ───────────────────────────────────────────────────────

def train_fold(
    fold_idx:   int,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    patch_dir:  Path = PROCESSED_DIR,
    device:     Optional[torch.device] = None,
    lr:         float = LR_DEFAULT,
    batch_size: int   = BATCH_SIZE,
    max_epochs: int   = MAX_EPOCHS,
    patience:   int   = PATIENCE,
    save_best:  bool  = True,
) -> dict:
    """
    Bir fold için EfficientNet-B0 eğitir.

    Parameters
    ----------
    fold_idx  : fold numarası (0-tabanlı)
    train_df  : eğitim hastalarının DataFrame'i (patient_id, pCR)
    val_df    : doğrulama hastalarının DataFrame'i
    patch_dir : .npy dosyalarının dizini
    save_best : True → en iyi val AUC'u outputs/models/best_cnn.pt'ye kaydeder

    Returns
    -------
    dict — fold metrikleri + eğitim geçmişi
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Fold %d | device=%s", fold_idx, device)

    # Dataset & DataLoader
    train_ds = PatchDataset(patch_dir, train_df, train=True)
    val_ds   = PatchDataset(patch_dir, val_df,   train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # Model
    model = build_efficientnet_b0(freeze_backbone=True).to(device)

    # Kayıp ve optimizer
    y_train_arr = train_df["pCR"].astype(int).values
    criterion   = build_criterion(y_train_arr, device)
    optimizer   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    early_stop = EarlyStopping(patience=patience)
    best_state: Optional[dict] = None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Eğitim döngüsü
    for epoch in range(1, max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, y_true, y_prob = validate_one_epoch(model, val_loader, criterion, device)

        val_auc = float(roc_auc_score(y_true, y_prob)) \
                  if len(np.unique(y_true)) > 1 else 0.5

        history["train_loss"].append(round(tr_loss, 5))
        history["val_loss"].append(round(vl_loss, 5))
        history["val_auc"].append(round(val_auc, 4))

        scheduler.step(val_auc)

        if val_auc > early_stop.best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "  Epoch %3d | tr_loss=%.4f  vl_loss=%.4f  val_AUC=%.3f  "
                "LR=%.2e  (best=%.3f  patience=%d/%d)",
                epoch, tr_loss, vl_loss, val_auc,
                optimizer.param_groups[0]["lr"],
                early_stop.best, early_stop._counter, patience,
            )

        if early_stop.step(val_auc):
            log.info("  Erken durdurma tetiklendi — epoch %d", epoch)
            break

    # En iyi ağırlıkları yükle
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final doğrulama metrikleri
    _, y_true, y_prob = validate_one_epoch(model, val_loader, criterion, device)
    metrics = compute_val_metrics(y_true, y_prob)
    metrics["fold"]          = fold_idx
    metrics["best_val_auc"]  = round(early_stop.best, 4)
    metrics["stopped_epoch"] = len(history["val_auc"])

    log.info(
        "Fold %d tamamlandi | AUC=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f  MCC=%.3f",
        fold_idx,
        metrics["auc_roc"], metrics["f1"],
        metrics["sensitivity"], metrics["specificity"], metrics["mcc"],
    )

    # Model kaydetme
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = MODELS_DIR / f"cnn_fold{fold_idx}.pt"
    torch.save(
        {
            "fold":     fold_idx,
            "state":    best_state,
            "metrics":  metrics,
            "history":  history,
        },
        fold_path,
    )

    if save_best and (
        not BEST_MODEL_PATH.exists()
        or metrics["auc_roc"] > _read_best_auc()
    ):
        torch.save({"state": best_state, "metrics": metrics}, BEST_MODEL_PATH)
        log.info("En iyi model guncellendi: AUC=%.3f  ->  %s",
                 metrics["auc_roc"], BEST_MODEL_PATH)

    return {"metrics": metrics, "history": history}


# ── 10. Tüm fold'lar CV ────────────────────────────────────────────────────────

def run_cv_training(
    labels_df:  pd.DataFrame,
    patch_dir:  Path = PROCESSED_DIR,
    n_splits:   int  = CV_FOLDS,
    **train_kwargs,
) -> dict:
    """
    5-fold GroupKFold CV ile tüm fold'ları eğitir.

    Returns
    -------
    {
      "fold_results": [fold_metrics, ...],
      "summary": {metric: {"mean": float, "std": float}, ...}
    }
    """
    _set_seed(RANDOM_SEED)

    splits, df_avail = create_fold_splits(labels_df, patch_dir, n_splits)
    all_metrics: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        log.info("=" * 58)
        log.info("FOLD %d / %d", fold_idx + 1, len(splits))
        log.info("=" * 58)

        train_df = df_avail.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_avail.iloc[val_idx].reset_index(drop=True)

        result = train_fold(
            fold_idx, train_df, val_df,
            patch_dir=patch_dir,
            **train_kwargs,
        )
        all_metrics.append(result["metrics"])

    summary = _summarize_cv(all_metrics)
    _print_cv_summary(summary)

    return {"fold_results": all_metrics, "summary": summary}


def _summarize_cv(fold_metrics: list[dict]) -> dict:
    metric_keys = ["auc_roc", "f1", "sensitivity", "specificity", "mcc"]
    summary: dict = {}
    for k in metric_keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(m.get(k, np.nan))]
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
        }
    return summary


def _print_cv_summary(summary: dict) -> None:
    sep = "-" * 52
    print(f"\n{sep}")
    print("  CNN 5-FOLD CV OZETI")
    print(sep)
    for k, v in summary.items():
        print(f"  {k:<14} {v['mean']:.3f} +/- {v['std']:.3f}")
    print(sep + "\n")


# ── 11. Sonuç kaydetme ────────────────────────────────────────────────────────

def save_cnn_results(cv_output: dict) -> Path:
    """CV sonuçlarını outputs/results/cnn_results.json olarak kaydeder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "cnn_results.json"

    payload = {
        "summary":      cv_output["summary"],
        "fold_results": cv_output["fold_results"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("CNN sonuclari kaydedildi: %s", path)
    return path


# ── 12. Checkpoint yardımcıları ───────────────────────────────────────────────

def load_best_model(device: Optional[torch.device] = None) -> nn.Module:
    """En iyi kaydedilmiş modeli yükler ve GradCAM için hazırlar."""
    if device is None:
        device = torch.device("cpu")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model bulunamadi: {BEST_MODEL_PATH}")

    ckpt  = torch.load(BEST_MODEL_PATH, map_location=device)
    model = build_efficientnet_b0(freeze_backbone=True).to(device)
    model.load_state_dict(ckpt["state"])
    model = unfreeze_for_gradcam(model)   # GradCAM için hazırlık
    model.eval()
    log.info("Model yuklendi: %s  (AUC=%.3f)",
             BEST_MODEL_PATH, ckpt["metrics"].get("auc_roc", float("nan")))
    return model


def _read_best_auc() -> float:
    try:
        ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
        return float(ckpt.get("metrics", {}).get("auc_roc", 0.0))
    except Exception:
        return 0.0


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="EfficientNet-B0 NAC yanit tahmini")
    parser.add_argument("--fold",       type=int,   default=None,
                        help="Sadece bu fold'u calistir (0-tabanli). None=hepsi.")
    parser.add_argument("--patch-dir",  default=str(PROCESSED_DIR))
    parser.add_argument("--lr",         type=float, default=LR_DEFAULT)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int,   default=MAX_EPOCHS)
    parser.add_argument("--patience",   type=int,   default=PATIENCE)
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)

    # Etiket tablosunu yükle (gerçek kullanımda data_loader.load_clinical())
    catalog_path = RESULTS_DIR / "merged_catalog.csv"
    if catalog_path.exists():
        labels_df = pd.read_csv(catalog_path)[["patient_id", "pCR"]].dropna()
    else:
        # Demo: sentetik etiketler
        log.warning("merged_catalog.csv bulunamadi; demo etiketleri kullaniliyor.")
        npy_files = sorted(patch_dir.glob("*.npy"))
        if not npy_files:
            raise SystemExit(
                "Hic .npy patch bulunamadi.\n"
                "Oncelikle: python harmonize.py"
            )
        rng = np.random.default_rng(RANDOM_SEED)
        labels_df = pd.DataFrame({
            "patient_id": [p.stem for p in npy_files],
            "pCR":        (rng.random(len(npy_files)) < 0.30).astype(int),
        })

    labels_df = labels_df.reset_index(drop=True)
    print(f"Veri: n={len(labels_df)}  pCR={labels_df['pCR'].sum()}")

    train_kwargs = dict(
        lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.epochs, patience=args.patience,
    )

    if args.fold is not None:
        splits, df_avail = create_fold_splits(labels_df, patch_dir)
        tr_idx, val_idx  = splits[args.fold]
        result = train_fold(
            args.fold,
            df_avail.iloc[tr_idx].reset_index(drop=True),
            df_avail.iloc[val_idx].reset_index(drop=True),
            patch_dir=patch_dir,
            **train_kwargs,
        )
        print(result["metrics"])
    else:
        cv_out = run_cv_training(labels_df, patch_dir, **train_kwargs)
        save_cnn_results(cv_out)
