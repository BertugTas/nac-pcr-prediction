"""
DCE-MRI Radyomik Özellik Çıkarıcı

Her .npy patch dosyasından (3, 128, 128) şu kategorilerde özellikler çıkarır:

  - intensity   : pre/early/late kanallarında yoğunluk istatistikleri (mean, std, skew…)
  - kinetic     : SER haritasından DCE kinetik göstergeler (PE, WR, SER istatistikleri)
  - glcm        : Gri-seviye birlikte oluşum matrisi doku özellikleri
  - lbp         : Yerel ikili örüntü histogramı özellikleri
  - roi_shape   : SER eşikli ROI'den şekil özellikleri (alan, çevre, yuvarlaklık)

Kullanım:
  python src/radiomic_features.py
  python src/radiomic_features.py --processed-dir data/processed --output outputs/results/radiomic_features.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from config import PROCESSED_DIR, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

SER_THRESHOLD   = 0.3   # ROI maskesi için SER eşiği
LBP_RADIUS      = 2
LBP_N_POINTS    = 8 * LBP_RADIUS
LBP_METHOD      = "uniform"
GLCM_DISTANCES  = [1, 2, 3]
GLCM_ANGLES     = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS      = ["contrast", "dissimilarity", "homogeneity", "energy",
                   "correlation", "ASM"]

EPS = 1e-6


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _safe_stats(arr: np.ndarray) -> dict[str, float]:
    """Temel istatistikler — boş veya sabit dizilerde güvenli."""
    arr = arr.ravel().astype(np.float64)
    if arr.size == 0:
        return {k: 0.0 for k in ("mean", "std", "skew", "kurt",
                                  "p10", "p25", "p50", "p75", "p90", "iqr")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "skew": float(stats.skew(arr)),
            "kurt": float(stats.kurtosis(arr)),
            "p10":  float(np.percentile(arr, 10)),
            "p25":  float(np.percentile(arr, 25)),
            "p50":  float(np.percentile(arr, 50)),
            "p75":  float(np.percentile(arr, 75)),
            "p90":  float(np.percentile(arr, 90)),
            "iqr":  float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Float görüntüyü [0,255] uint8'e dönüştür."""
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < EPS:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


# ── Özellik çıkarıcılar ───────────────────────────────────────────────────────

def extract_intensity_features(patch: np.ndarray) -> dict[str, float]:
    """
    pre / early / late kanallarından yoğunluk istatistikleri.
    patch shape: (3, H, W)
    """
    feats: dict[str, float] = {}
    for i, name in enumerate(("pre", "early", "late")):
        s = _safe_stats(patch[i])
        feats.update({f"int_{name}_{k}": v for k, v in s.items()})
    return feats


def extract_kinetic_features(patch: np.ndarray) -> dict[str, float]:
    """
    SER haritasından DCE kinetik özellikler.
      PE   = peak enhancement = mean(early - pre)
      WR   = washout ratio    = mean(early - late) / (mean(early - pre) + eps)
      SER  = signal enhancement ratio = (early-pre) / (|late-pre| + eps)
    """
    pre, early, late = patch[0], patch[1], patch[2]

    pe_map  = early - pre
    ser_map = pe_map / (np.abs(late - pre) + EPS)
    ser_map = np.where(pe_map > 0, ser_map, 0.0)

    feats: dict[str, float] = {}

    # SER istatistikleri
    s = _safe_stats(ser_map)
    feats.update({f"kin_ser_{k}": v for k, v in s.items()})

    # PE istatistikleri
    s = _safe_stats(pe_map)
    feats.update({f"kin_pe_{k}": v for k, v in s.items()})

    # Kinetik göstergeler
    mean_pe   = float(np.mean(pe_map))
    mean_wr   = float(np.mean(early - late))
    feats["kin_peak_enhancement"]  = mean_pe
    feats["kin_washout_ratio"]     = mean_wr / (mean_pe + EPS)
    feats["kin_ser_positive_frac"] = float(np.mean(ser_map > SER_THRESHOLD))
    feats["kin_ser_max"]           = float(np.max(ser_map))
    feats["kin_ser_p95"]           = float(np.percentile(ser_map, 95))

    # Türetilmiş özellikler: tümör hacmi, ham washout, kontrast artış eğimi
    feats["kin_tumor_volume"]  = float(np.sum(ser_map > SER_THRESHOLD))
    feats["kin_washout_raw"]   = float(np.mean(late / (early + EPS)))
    feats["kin_ce_slope"]      = float(np.mean(pe_map / (np.abs(pre) + EPS)))

    return feats


def extract_glcm_features(patch: np.ndarray) -> dict[str, float]:
    """
    GLCM doku özellikleri — early post-contrast kanalından.
    """
    img_u8 = _to_uint8(patch[1])   # early phase

    feats: dict[str, float] = {}
    try:
        glcm = graycomatrix(
            img_u8,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )
        for prop in GLCM_PROPS:
            vals = graycoprops(glcm, prop)   # shape (len(distances), len(angles))
            feats[f"glcm_{prop}_mean"] = float(vals.mean())
            feats[f"glcm_{prop}_std"]  = float(vals.std())
    except Exception as exc:
        log.debug("GLCM hesaplanamadi: %s", exc)
        for prop in GLCM_PROPS:
            feats[f"glcm_{prop}_mean"] = 0.0
            feats[f"glcm_{prop}_std"]  = 0.0

    return feats


def extract_lbp_features(patch: np.ndarray) -> dict[str, float]:
    """
    LBP histogram özellikleri — pre ve early kanallarından.
    """
    feats: dict[str, float] = {}
    for i, name in enumerate(("pre", "early")):
        img_u8 = _to_uint8(patch[i])
        lbp = local_binary_pattern(
            img_u8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD
        )
        n_bins = LBP_N_POINTS + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                                range=(0, n_bins), density=True)
        feats.update({f"lbp_{name}_bin{j}": float(hist[j]) for j in range(n_bins)})
        feats[f"lbp_{name}_entropy"] = float(
            -np.sum(hist * np.log(hist + EPS))
        )
    return feats


def extract_roi_shape_features(patch: np.ndarray) -> dict[str, float]:
    """
    SER > threshold ile oluşturulan ROI maskesinden şekil özellikleri.
    """
    pre, early, late = patch[0], patch[1], patch[2]
    pe_map  = early - pre
    ser_map = pe_map / (np.abs(late - pre) + EPS)
    ser_map = np.where(pe_map > 0, ser_map, 0.0)

    mask   = (ser_map > SER_THRESHOLD).astype(np.uint8)
    lbl    = label(mask)
    feats: dict[str, float] = {}

    if lbl.max() == 0:
        return {
            "shape_roi_area": 0.0, "shape_roi_perimeter": 0.0,
            "shape_compactness": 0.0, "shape_eccentricity": 0.0,
            "shape_solidity": 0.0, "shape_major_axis": 0.0,
            "shape_minor_axis": 0.0, "shape_extent": 0.0,
        }

    # En büyük bileşeni al
    regions = sorted(regionprops(lbl), key=lambda r: r.area, reverse=True)
    r = regions[0]

    area = float(r.area)
    perim = float(r.perimeter) if r.perimeter > 0 else 1.0
    feats["shape_roi_area"]       = area
    feats["shape_roi_perimeter"]  = perim
    feats["shape_compactness"]    = (4 * np.pi * area) / (perim ** 2 + EPS)
    feats["shape_eccentricity"]   = float(r.eccentricity)
    feats["shape_solidity"]       = float(r.solidity)
    feats["shape_major_axis"]     = float(r.axis_major_length)
    feats["shape_minor_axis"]     = float(r.axis_minor_length)
    feats["shape_extent"]         = float(r.extent)

    return feats


# ── Tek patch için tam özellik çıkarıcı ──────────────────────────────────────

def extract_all_features(patch: np.ndarray, patient_id: str) -> dict:
    """(3, H, W) patch'ten tüm özellikleri çıkarır."""
    feats: dict = {"patient_id": patient_id}

    feats.update(extract_intensity_features(patch))
    feats.update(extract_kinetic_features(patch))
    feats.update(extract_glcm_features(patch))
    feats.update(extract_lbp_features(patch))
    feats.update(extract_roi_shape_features(patch))

    return feats


# ── Toplu çıkarıcı ───────────────────────────────────────────────────────────

def extract_radiomic_features(
    processed_dir: Path = PROCESSED_DIR,
    output_path:   Path = RESULTS_DIR / "radiomic_features.csv",
    catalog_path:  Path = RESULTS_DIR / "merged_catalog.csv",
) -> pd.DataFrame:
    """
    data/processed/ altındaki tüm .npy patch'lerden özellik çıkarır,
    merged_catalog.csv ile birleştirir ve CSV olarak kaydeder.
    """
    npy_files = sorted(processed_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"data/processed/ altında .npy dosyası bulunamadı: {processed_dir}")

    log.info("%d patch dosyası bulundu.", len(npy_files))

    records: list[dict] = []
    n_ok, n_fail = 0, 0

    for npy_path in npy_files:
        pid = npy_path.stem   # dosya adı = patient_id
        if pid.endswith("_meta"):
            continue
        try:
            patch = np.load(str(npy_path)).astype(np.float32)
            if patch.ndim != 3 or patch.shape[0] != 3:
                log.warning("[%s] Beklenen (3,H,W) degil: %s — atlaniyor.", pid, patch.shape)
                n_fail += 1
                continue
            feats = extract_all_features(patch, pid)
            records.append(feats)
            n_ok += 1
        except Exception as exc:
            log.warning("[%s] Ozellik cikarilamadi: %s", pid, exc)
            n_fail += 1

    if not records:
        raise ValueError("Hiçbir patch'ten özellik çıkarılamadı.")

    df = pd.DataFrame(records)
    n_features = df.shape[1] - 1   # patient_id hariç

    log.info("Ozellik cikarma tamamlandi: %d OK / %d hata / hasta basi %d ozellik",
             n_ok, n_fail, n_features)

    # Klinik verilerle birleştir
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path, dtype=str)
        catalog["patient_id"] = catalog["patient_id"].astype(str).str.strip()
        df["patient_id"]      = df["patient_id"].astype(str).str.strip()
        df = pd.merge(df, catalog, on="patient_id", how="left")
        log.info("Klinik katalogla birlestirildi: %d hasta.", len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Kaydedildi: %s", output_path)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCE-MRI radyomik ozellik cikarici")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--output",        default=str(RESULTS_DIR / "radiomic_features.csv"))
    parser.add_argument("--catalog",       default=str(RESULTS_DIR / "merged_catalog.csv"))
    args = parser.parse_args()

    df = extract_radiomic_features(
        processed_dir=Path(args.processed_dir),
        output_path=Path(args.output),
        catalog_path=Path(args.catalog),
    )

    # Özet
    n_feat = sum(1 for c in df.columns if c not in ("patient_id",))
    pcr_col = "pCR" if "pCR" in df.columns else None
    print(f"\n{'='*55}")
    print(f"  Toplam hasta          : {len(df)}")
    print(f"  Radyomik özellik sayısı: {n_feat}")
    print(f"  Özellik kategorileri  : intensity, kinetic, glcm, lbp, roi_shape")
    if pcr_col:
        labeled = df[pcr_col].notna()
        print(f"  pCR etiketi olan hasta: {labeled.sum()} / {len(df)}")
    print(f"{'='*55}")
