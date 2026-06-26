"""
3D radyomik özellik çıkarıcı — tam tümör hacminden (pyradiomics gerektirmez).

Her hasta için _meta.json'daki DICOM yollarından tam 3D hacim yüklenir,
SER eşiği ile tümör maskesi oluşturulur ve şu kategorilerde özellikler çıkarılır:

  fo3_*    : 3D first-order istatistikler (early-post kontrast + SER)
  glcm3_*  : 3D gray-level co-occurrence matrix (13 yön, ortalamalı)
  rlm3_*   : 3D gray-level run-length matrix (3 eksen, ortalamalı)
  shape3_* : 3D şekil özellikleri (hacim, yüzey alanı, küresellik, …)

Kullanım:
  python src/radiomic_features_3d.py   # tüm hastalar, cache'e yazar
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats
from skimage.measure import label, marching_cubes

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from config import PROCESSED_DIR, RESULTS_DIR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")

# ── Sabitler ──────────────────────────────────────────────────────────────────
EPS           = 1e-6
SER_THRESHOLD = 0.3      # tümör maskesi için SER eşiği
N_GRAY        = 32       # GLCM / GLRLM gri seviyesi
BBOX_MARGIN   = 15       # tümör sınırı etrafındaki ek voksel payı
MAX_RUN       = 50       # GLRLM için maksimum koşu uzunluğu

# 13 benzersiz 3D yön vektörü (yarım-uzay)
_OFFSETS_3D = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _load_series(series_dir: str) -> np.ndarray:
    """DICOM seri klasöründen (Z, H, W) float32 hacim yükler."""
    reader = sitk.ImageSeriesReader()
    files  = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        # Düz .dcm dosyaları (manifest yapısı dışı)
        files = sorted(str(p) for p in Path(series_dir).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"DICOM bulunamadı: {series_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()
    # LPS orientasyonu
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("LPS")
    img = orient.Execute(img)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _match_shape(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Hacmi hedef şekle kırpar veya sıfır-pads."""
    sl = tuple(slice(0, min(vs, ts)) for vs, ts in zip(vol.shape, target))
    crop = vol[sl]
    result = np.zeros(target, dtype=np.float32)
    result[sl] = crop
    return result


def _normalize_percentile(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    return np.clip((vol - lo) / (hi - lo + EPS), 0, 1).astype(np.float32)


def _quantize(vol: np.ndarray, n_gray: int = N_GRAY) -> np.ndarray:
    q = np.clip(vol * (n_gray - 1), 0, n_gray - 1)
    return q.astype(np.int32)


def _crop_to_mask(vol: np.ndarray, mask: np.ndarray,
                  margin: int = BBOX_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """Maskenin bounding box'una ±margin voksel ekleyerek kırp."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return vol, mask
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, np.array(vol.shape))
    sl = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
    return vol[sl], mask[sl]


# ── 3D First-order özellikler ─────────────────────────────────────────────────

def _firstorder_3d(vol: np.ndarray, mask: np.ndarray,
                   prefix: str = "fo3") -> dict[str, float]:
    vx = vol[mask > 0].astype(np.float64)
    if vx.size == 0:
        keys = ["mean", "std", "skew", "kurt", "energy",
                "entropy", "rms", "uniformity", "p10", "p90", "iqr"]
        return {f"{prefix}_{k}": 0.0 for k in keys}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, _ = np.histogram(vx, bins=64)
    p = hist / (hist.sum() + EPS)

    return {
        f"{prefix}_mean":        float(np.mean(vx)),
        f"{prefix}_std":         float(np.std(vx)),
        f"{prefix}_skew":        float(stats.skew(vx)),
        f"{prefix}_kurt":        float(stats.kurtosis(vx)),
        f"{prefix}_energy":      float(np.sum(vx ** 2)),
        f"{prefix}_entropy":     float(-np.sum(p * np.log2(p + EPS))),
        f"{prefix}_rms":         float(np.sqrt(np.mean(vx ** 2))),
        f"{prefix}_uniformity":  float(np.sum(p ** 2)),
        f"{prefix}_p10":         float(np.percentile(vx, 10)),
        f"{prefix}_p90":         float(np.percentile(vx, 90)),
        f"{prefix}_iqr":         float(np.percentile(vx, 75) - np.percentile(vx, 25)),
    }


# ── 3D GLCM özellikler ────────────────────────────────────────────────────────

def _glcm_3d(vol_q: np.ndarray, mask: np.ndarray,
             n_gray: int = N_GRAY) -> dict[str, float]:
    """13 yön üzerinden ortalanmış 3D GLCM."""
    glcm = np.zeros((n_gray, n_gray), dtype=np.float64)
    Z, H, W = vol_q.shape

    for dz, dy, dx in _OFFSETS_3D:
        z0, z1 = max(0, -dz), min(Z, Z - dz)
        y0, y1 = max(0, -dy), min(H, H - dy)
        x0, x1 = max(0, -dx), min(W, W - dx)

        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        src = vol_q[z0:z1, y0:y1, x0:x1]
        tgt = vol_q[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        m0  = mask[z0:z1, y0:y1, x0:x1]
        m1  = mask[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]

        valid = (m0 > 0) & (m1 > 0)
        if not valid.any():
            continue
        gs, gt = src[valid].ravel(), tgt[valid].ravel()
        np.add.at(glcm, (gs, gt), 1)
        np.add.at(glcm, (gt, gs), 1)

    total = glcm.sum()
    if total < EPS:
        return {k: 0.0 for k in [
            "glcm3_contrast", "glcm3_correlation",
            "glcm3_energy", "glcm3_homogeneity",
            "glcm3_dissimilarity", "glcm3_asm",
        ]}

    p = glcm / total
    i_idx, j_idx = np.mgrid[0:n_gray, 0:n_gray]
    mu_i  = (p * i_idx).sum()
    mu_j  = (p * j_idx).sum()
    sig_i = np.sqrt((p * (i_idx - mu_i) ** 2).sum() + EPS)
    sig_j = np.sqrt((p * (j_idx - mu_j) ** 2).sum() + EPS)

    return {
        "glcm3_contrast":     float((p * (i_idx - j_idx) ** 2).sum()),
        "glcm3_correlation":  float((p * (i_idx - mu_i) * (j_idx - mu_j)).sum()
                                    / (sig_i * sig_j)),
        "glcm3_energy":       float(np.sqrt((p ** 2).sum())),
        "glcm3_homogeneity":  float((p / (1 + np.abs(i_idx - j_idx))).sum()),
        "glcm3_dissimilarity":float((p * np.abs(i_idx - j_idx)).sum()),
        "glcm3_asm":          float((p ** 2).sum()),
    }


# ── 3D GLRLM özellikler ───────────────────────────────────────────────────────

def _glrlm_axis(vol_q: np.ndarray, mask: np.ndarray,
                axis: int, n_gray: int = N_GRAY,
                max_run: int = MAX_RUN) -> np.ndarray:
    """Tek eksende (z/y/x) gray-level run-length matrix."""
    vol_t  = np.moveaxis(vol_q, axis, 0)   # (L, *, *)
    mask_t = np.moveaxis(mask,  axis, 0)
    L, H, W = vol_t.shape
    rlm = np.zeros((n_gray, max_run + 1), dtype=np.float64)

    for j in range(H):
        for k in range(W):
            line = vol_t[:, j, k]
            m    = mask_t[:, j, k]
            pos  = np.where(m > 0)[0]
            if pos.size == 0:
                continue

            run_len = 1
            prev_g  = int(line[pos[0]])
            prev_i  = pos[0]
            for idx in pos[1:]:
                g = int(line[idx])
                if g == prev_g and idx == prev_i + 1:
                    run_len += 1
                else:
                    rlm[prev_g, min(run_len, max_run)] += 1
                    run_len = 1
                    prev_g  = g
                prev_i = idx
            rlm[prev_g, min(run_len, max_run)] += 1

    return rlm


def _rlm_stats(rlm: np.ndarray, n_gray: int = N_GRAY) -> dict[str, float]:
    total = rlm.sum()
    if total < EPS:
        return {k: 0.0 for k in [
            "rlm3_sre", "rlm3_lre", "rlm3_lgre",
            "rlm3_hgre", "rlm3_rlnu", "rlm3_glnu", "rlm3_rp",
        ]}
    p = rlm / total
    i_idx = np.arange(n_gray)[:, None]          # gray level
    j_idx = np.arange(rlm.shape[1])[None, :]    # run length
    js    = np.maximum(j_idx, 1)                 # avoid /0

    ri = rlm.sum(axis=1)
    rj = rlm.sum(axis=0)

    return {
        "rlm3_sre":  float((p / (js ** 2)).sum()),
        "rlm3_lre":  float((p * j_idx ** 2).sum()),
        "rlm3_lgre": float((p / (i_idx ** 2 + EPS)).sum()),
        "rlm3_hgre": float((p * i_idx ** 2).sum()),
        "rlm3_rlnu": float((rj ** 2).sum() / total),
        "rlm3_glnu": float((ri ** 2).sum() / total),
        "rlm3_rp":   float(total / max(rlm.size, 1)),
    }


def _glrlm_3d(vol_q: np.ndarray, mask: np.ndarray,
              n_gray: int = N_GRAY) -> dict[str, float]:
    """3 eksen üzerinden ortalanmış GLRLM."""
    all_feats = []
    for ax in range(3):
        rlm  = _glrlm_axis(vol_q, mask, ax, n_gray)
        all_feats.append(_rlm_stats(rlm, n_gray))
    keys = list(all_feats[0].keys())
    return {k: float(np.mean([f[k] for f in all_feats])) for k in keys}


# ── 3D şekil özellikleri ─────────────────────────────────────────────────────

def _shape_3d(mask: np.ndarray) -> dict[str, float]:
    mb     = mask.astype(bool)
    volume = float(mb.sum())
    feats  = {"shape3_volume": volume}

    if volume < 8:
        for k in ["shape3_surface_area", "shape3_sphericity",
                  "shape3_compactness", "shape3_elongation", "shape3_flatness"]:
            feats[k] = 0.0
        return feats

    # Yüzey alanı (marching cubes)
    try:
        verts, faces, _, _ = marching_cubes(mb.astype(np.float32), level=0.5)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        sa = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
    except Exception:
        sa = float(volume ** (2 / 3)) * 6.0   # kaba küp tahmini

    feats["shape3_surface_area"] = sa
    feats["shape3_sphericity"]   = float(
        (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / (sa + EPS)
    )
    feats["shape3_compactness"]  = float(volume / ((sa ** 1.5) + EPS))

    # PCA ile uzama / düzlük
    coords = np.argwhere(mb).astype(float)
    if len(coords) >= 3:
        coords -= coords.mean(axis=0)
        cov     = np.cov(coords.T)
        eigs    = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
        e1, e2, e3 = eigs[0] + EPS, eigs[1] + EPS, eigs[2] + EPS
        feats["shape3_elongation"] = float(e2 / e1)
        feats["shape3_flatness"]   = float(e3 / e1)
    else:
        feats["shape3_elongation"] = 0.0
        feats["shape3_flatness"]   = 0.0

    return feats


# ── Tek hasta çıkarımı ───────────────────────────────────────────────────────

def extract_patient_3d(pid: str, meta_path: Path) -> dict:
    """
    Bir hastanın _meta.json'ından DICOM yollarını okur,
    tam 3D hacimleri yükler ve tüm 3D özellikleri döner.
    """
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Başarısız harmonizasyonu atla
    if meta.get("status") == "failed":
        raise ValueError("Harmonizasyon başarısız olmuş hasta.")

    phases = meta.get("phases_used", {})
    pre_p, early_p, late_p = phases.get("pre"), phases.get("early"), phases.get("late")
    if not all([pre_p, early_p, late_p]):
        raise ValueError("meta.json'da eksik faz yolları.")

    # Yükle
    pre   = _load_series(pre_p)
    early = _load_series(early_p)
    late  = _load_series(late_p)

    target = pre.shape
    early  = _match_shape(early, target)
    late   = _match_shape(late,  target)

    # Normalize
    pre_n   = _normalize_percentile(pre)
    early_n = _normalize_percentile(early)
    late_n  = _normalize_percentile(late)

    # SER haritası + tümör maskesi
    pe_map  = early_n - pre_n
    ser_map = pe_map / (np.abs(late_n - pre_n) + EPS)
    ser_map = np.where(pe_map > 0, ser_map, 0.0)
    mask    = (ser_map > SER_THRESHOLD).astype(np.uint8)

    if mask.sum() < 8:
        # Daha düşük eşik dene
        mask = (ser_map > SER_THRESHOLD * 0.5).astype(np.uint8)
    if mask.sum() < 8:
        raise ValueError(f"Tümör maskesi çok küçük: {mask.sum()} voksel.")

    # Bounding box kırpma (GLCM/GLRLM hızı için)
    early_c, mask_c = _crop_to_mask(early_n, mask)
    ser_c,   _      = _crop_to_mask(ser_map, mask)

    vol_q = _quantize(early_c, N_GRAY)

    feats: dict = {"patient_id": pid}
    feats.update(_firstorder_3d(early_c, mask_c, prefix="fo3_early"))
    feats.update(_firstorder_3d(ser_c,   (ser_c > 0).astype(np.uint8), prefix="fo3_ser"))
    feats.update(_glcm_3d(vol_q, mask_c, N_GRAY))
    feats.update(_glrlm_3d(vol_q, mask_c, N_GRAY))
    feats.update(_shape_3d(mask_c))

    return feats


# ── Toplu çıkarım + önbellek ─────────────────────────────────────────────────

def extract_3d_features_all_patients(
    processed_dir: Path = PROCESSED_DIR,
    output_path:   Path = RESULTS_DIR / "radiomic_features_3d.csv",
    patient_ids:   Optional[set] = None,
    force:         bool = False,
) -> pd.DataFrame:
    """
    Tüm hastalar için 3D özellikleri çıkarır ve CSV'ye kaydeder.
    Cache dosyası mevcutsa yeniden yüklenir (force=True ile yeniden hesaplanır).

    Parameters
    ----------
    patient_ids : işlenecek hasta ID seti (None → meta.json olan hepsi)
    force       : True ise cache'i görmezden gel ve yeniden hesapla
    """
    output_path = Path(output_path)

    if output_path.exists() and not force:
        log.info("3D özellik cache yükleniyor: %s", output_path)
        return pd.read_csv(output_path, dtype=str)

    meta_files: dict[str, Path] = {
        p.stem.replace("_meta", ""): p
        for p in sorted(Path(processed_dir).glob("*_meta.json"))
        if p.stat().st_size > 100
    }
    if patient_ids is not None:
        meta_files = {k: v for k, v in meta_files.items() if k in patient_ids}

    log.info("3D radyomik: %d hasta işlenecek …", len(meta_files))
    records: list[dict] = []
    n_ok = n_fail = 0

    for pid, meta_path in sorted(meta_files.items()):
        try:
            feats = extract_patient_3d(pid, meta_path)
            records.append(feats)
            n_ok += 1
            if n_ok % 20 == 0:
                log.info("  3D: %d / %d tamamlandı.", n_ok, len(meta_files))
        except Exception as exc:
            log.debug("[%s] 3D hata: %s", pid, exc)
            n_fail += 1

    log.info("3D radyomik bitti: %d OK, %d hata.", n_ok, n_fail)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    log.info("3D özellikler kaydedildi: %s (%d hasta, %d sütun)",
             output_path, len(df), df.shape[1])
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="3D radyomik özellik çıkarıcı")
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--output", default=str(RESULTS_DIR / "radiomic_features_3d.csv"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = extract_3d_features_all_patients(
        processed_dir=Path(args.processed_dir),
        output_path=Path(args.output),
        force=args.force,
    )
    if not df.empty:
        print(f"\n3D özellik matrisi: {df.shape[0]} hasta × {df.shape[1]-1} özellik")
        print("Örnek sütunlar:", df.columns[1:6].tolist())
