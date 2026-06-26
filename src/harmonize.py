"""
Çok-merkezli DCE-MRI harmonizasyon hattı.

Adımlar (her hasta için):
  1. DICOM yükle  →  sitk.Image
  2. Orientasyon düzelt  →  axial LPS
  3. Voxel spacing'i 1×1×1 mm'ye resample et
  4. Yoğunluk normalize et  (percentile clip + z-score + alan gücü dengesi)
  5. SER haritasından ROI merkezi bul
  6. 128×128 px patch çıkar, 3 faz [3, H, W] olarak birleştir
  7. .npy + JSON kaydet

Komut satırından çalıştırmak için:
  python harmonize.py                  # tüm ISPY1 hastaları
  python harmonize.py --patient P001   # tek hasta
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from config import PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

ISPY1_DIR     = RAW_DIR / "ISPY1"
TARGET_SPACING = (1.0, 1.0, 1.0)   # mm
PATCH_SIZE     = 128                # piksel
PERCENTILE_LO  = 1.0
PERCENTILE_HI  = 99.0
SER_THRESHOLD  = 0.5                # SER piksel seçim eşiği (ROI için)

# 1.5T → 3T sinyal ölçekleme tahmini (literatür tabanlı kaba düzeltme)
_FIELD_STRENGTH_SCALE: dict[float, float] = {
    1.5: 1.30,
    3.0: 1.00,
}

# DCE faz tanımlama regex'leri  (pre < early-post < late-post)
_PHASE_PATTERNS: dict[str, re.Pattern] = {
    "pre":   re.compile(r"pre|phase[\s_-]?0|t0|baseline|unenhanced", re.I),
    "early": re.compile(r"ph[\s_-]?1|post[\s_-]?1|first[\s_-]?post|early|1st|pe[\s_-]?1\b|pe1\b", re.I),
    "late":  re.compile(r"ph[\s_-]?[234]|post[\s_-]?[234]|late|delayed|last|pe[\s_-]?[234]\b|pe[234]\b|ser\b", re.I),
}


# ── Veri yapıları ─────────────────────────────────────────────────────────────

@dataclass
class PhaseSet:
    """Bir hastanın pre / early-post / late-post seri dizinleri."""
    pre:   Optional[Path] = None
    early: Optional[Path] = None
    late:  Optional[Path] = None

    def is_complete(self) -> bool:
        return all([self.pre, self.early, self.late])

    def missing(self) -> list[str]:
        return [k for k, v in asdict(self).items() if v is None]


@dataclass
class PatientMeta:
    patient_id:    str
    field_strength: float
    original_spacing: tuple
    resampled_spacing: tuple
    orientation_before: str
    orientation_after:  str
    tumor_center_voxel: tuple
    patch_size:    int
    phases_used:   dict
    n_slices_pre:  int
    status:        str = "ok"
    error:         str = ""


# ── 1. Yoğunluk normalizasyonu ────────────────────────────────────────────────

def percentile_clip(volume: np.ndarray,
                    lo: float = PERCENTILE_LO,
                    hi: float = PERCENTILE_HI) -> np.ndarray:
    """Yağ/hava piksel uçlarını kırpar."""
    p_lo = np.percentile(volume, lo)
    p_hi = np.percentile(volume, hi)
    return np.clip(volume, p_lo, p_hi).astype(np.float32)


def zscore_normalize(volume: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score normalizasyonu.
    mask verilirse yalnızca doku piksellerinden istatistik hesaplar
    (arka plan sıfırları ortalama/std'yi çarptırır).
    """
    if mask is not None and mask.any():
        vals = volume[mask > 0]
    else:
        vals = volume[volume > volume.min()]   # hava voksellerini dışla

    mu, sigma = float(vals.mean()), float(vals.std())
    if sigma < 1e-6:
        return (volume - mu).astype(np.float32)
    return ((volume - mu) / sigma).astype(np.float32)


def field_strength_correction(volume: np.ndarray,
                               field_strength: float) -> np.ndarray:
    """
    1.5T ile 3T arasındaki SNR farkını kaba bir ölçeklemeyle dengeler.
    Gerçek projede ComBat (neuroCombat) tercih edilmeli.
    """
    scale = _FIELD_STRENGTH_SCALE.get(round(field_strength * 2) / 2, 1.0)
    return (volume * scale).astype(np.float32)


def normalize_volume(volume: np.ndarray,
                     field_strength: float = 3.0,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Tam normalizasyon hattı: clip → alan gücü → z-score."""
    volume = percentile_clip(volume)
    volume = field_strength_correction(volume, field_strength)
    volume = zscore_normalize(volume, mask)
    return volume


# ── 2. Orientasyon düzeltme ───────────────────────────────────────────────────

def get_orientation_code(sitk_image: sitk.Image) -> str:
    """Görüntünün mevcut anatomik orientasyon kodunu döner (ör. 'LPS', 'RAS')."""
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        sitk_image.GetDirection()
    )


def reorient_to_axial_lps(sitk_image: sitk.Image) -> sitk.Image:
    """
    Görüntüyü axial LPS (Left-Posterior-Superior) koordinat sistemine döndürür.
    ISPY1 sagittal serileri otomatik olarak axial'e çevrilir.
    LPS: DICOM standardı; X=Sol, Y=Arka, Z=Üst.
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("LPS")
    return orient_filter.Execute(sitk_image)


# ── 3. Voxel spacing resampling ───────────────────────────────────────────────

def resample_to_spacing(sitk_image: sitk.Image,
                        target_spacing: tuple = TARGET_SPACING,
                        interpolator=sitk.sitkBSpline) -> sitk.Image:
    """
    Görüntüyü hedef voxel spacing'e (mm) yeniden örnekler.
    Varsayılan: 1×1×1 mm izotropik.
    """
    orig_spacing = np.array(sitk_image.GetSpacing())          # (sx, sy, sz)
    orig_size    = np.array(sitk_image.GetSize())             # (nx, ny, nz)
    target_sp    = np.array(target_spacing)

    new_size = np.round(orig_size * orig_spacing / target_sp).astype(int)
    new_size = [max(1, int(s)) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(sitk_image.GetPixelIDValue()))

    return resampler.Execute(sitk_image)


# ── 4. SER haritası ve ROI ────────────────────────────────────────────────────

def compute_ser(pre: np.ndarray,
                early: np.ndarray,
                late: np.ndarray,
                eps: float = 1e-6) -> np.ndarray:
    """
    Signal Enhancement Ratio (SER) hesaplar.

    SER = (S_early - S_pre) / (S_late - S_pre + eps)

    Yüksek SER → erken tutulum, yavaş yıkanma → şüpheli malign bölge.
    Negatif değerler sıfırlanır.
    """
    ser = (early - pre) / (np.abs(late - pre) + eps)
    ser = np.where((early - pre) > 0, ser, 0.0)   # negatif uptake maskelenir
    return ser.astype(np.float32)


def find_tumor_center(ser: np.ndarray,
                      threshold: float = SER_THRESHOLD) -> tuple[int, int, int]:
    """
    SER haritasında eşik üstündeki piksellerin ağırlıklı merkezini bulur.
    Eşik üstünde piksel yoksa maksimum SER noktasını döner.

    Returns
    -------
    (z, y, x) voxel koordinatı
    """
    mask = ser >= threshold
    if mask.any():
        coords = np.argwhere(mask)
        weights = ser[mask]
        center = np.average(coords, axis=0, weights=weights)
        return tuple(int(round(c)) for c in center)  # type: ignore[return-value]

    log.debug("SER esigi uzerinde piksel yok; maksimum nokta kullaniliyor.")
    return tuple(int(c) for c in np.unravel_index(ser.argmax(), ser.shape))  # type: ignore[return-value]


def extract_patch(volume: np.ndarray,
                  center: tuple[int, int, int],
                  patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Hacimden (Z, H, W) en yüksek SER dilimine ortalanmış 2-D patch çıkarır.

    Parameters
    ----------
    volume     : (Z, H, W) array
    center     : (z, y, x) merkez voxel
    patch_size : çıkarılacak kare patch kenarı (piksel)

    Returns
    -------
    np.ndarray, şekil (patch_size, patch_size)
    """
    z, cy, cx = center
    z  = int(np.clip(z,  0, volume.shape[0] - 1))
    cy = int(np.clip(cy, 0, volume.shape[1] - 1))
    cx = int(np.clip(cx, 0, volume.shape[2] - 1))

    half = patch_size // 2
    slice_2d = volume[z]                     # (H, W)

    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = y0 + patch_size
    x1 = x0 + patch_size

    # Görüntü sınırı taşmasını düzelt
    if y1 > slice_2d.shape[0]:
        y1 = slice_2d.shape[0]
        y0 = max(0, y1 - patch_size)
    if x1 > slice_2d.shape[1]:
        x1 = slice_2d.shape[1]
        x0 = max(0, x1 - patch_size)

    patch = slice_2d[y0:y1, x0:x1]

    # Kenar hastaları için pad
    if patch.shape != (patch_size, patch_size):
        pad_h = patch_size - patch.shape[0]
        pad_w = patch_size - patch.shape[1]
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="reflect")

    return patch.astype(np.float32)


def build_rgb_patch(pre_vol:   np.ndarray,
                    early_vol: np.ndarray,
                    late_vol:  np.ndarray,
                    center:    tuple[int, int, int],
                    patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    3 DCE fazından [3, H, W] RGB benzeri tensor oluşturur.
      Kanal 0 : pre-contrast
      Kanal 1 : early post-contrast
      Kanal 2 : late post-contrast

    Returns
    -------
    np.ndarray, şekil (3, patch_size, patch_size), dtype float32
    """
    patches = [
        extract_patch(vol, center, patch_size)
        for vol in (pre_vol, early_vol, late_vol)
    ]
    return np.stack(patches, axis=0)   # (3, H, W)


# ── 5. ISPY1 faz keşif motoru ─────────────────────────────────────────────────

def _load_sitk_series(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not names:
        raise FileNotFoundError(f"DICOM dosyasi bulunamadi: {series_dir}")
    reader.SetFileNames(names)
    return reader.Execute()


def _get_field_strength(series_dir: Path) -> float:
    """İlk DICOM dosyasından manyetik alan gücünü okur."""
    import pydicom
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(series_dir.glob("**/*.dcm"))
    if not dcm_files:
        return 3.0
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        return float(getattr(ds, "MagneticFieldStrength", 3.0))
    except Exception:
        return 3.0


def _classify_phase(description: str) -> Optional[str]:
    for phase, pattern in _PHASE_PATTERNS.items():
        if pattern.search(description):
            return phase
    return None


def discover_dce_phases(patient_dir: Path) -> PhaseSet:
    """
    Bir hasta klasöründe tüm çalışmaları/serileri tarar ve
    pre / early / late fazlarını eşleştirir.

    Strateji (çalışma bazında, öncelik sırası):
      1. SeriesDescription/klasör adında açık faz etiketi (PE1, SER, vb.)
      2. DCE filtresi (dce/dyn/spgr/contrast/...) ile tanınan seriler
      3. Etiketlenen fazlarla aynı çalışmadaki en küçük seri numarası → pre
      4. Seri numarası sıralaması ile 3+ etiketsiz seri → ilk=pre, orta=early, son=late

    Returns
    -------
    PhaseSet — eksik fazlar None olarak kalır
    """
    import pydicom

    _DCE_RE = re.compile(
        r"dce|dynamic|dyn|contrast|ir3dfgre|3dfgre|thrive|spgr|grx", re.I
    )

    def _read_series_meta(series_dir: Path):
        """Returns (desc, ser_num) or None if not MR or no DICOMs."""
        dcm_files = sorted(series_dir.glob("*.dcm"))
        if not dcm_files:
            dcm_files = sorted(series_dir.glob("**/*.dcm"))
        if not dcm_files:
            return None
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            if getattr(ds, "Modality", "MR") != "MR":
                return None
            desc    = str(getattr(ds, "SeriesDescription", series_dir.name))
            ser_num = int(getattr(ds, "SeriesNumber", 9999))
            return desc, ser_num
        except Exception:
            return series_dir.name, 9999

    best: Optional[PhaseSet] = None

    for study_dir in sorted(patient_dir.iterdir()):
        if not study_dir.is_dir():
            continue

        # Çalışmadaki tüm MR serilerini tara
        all_series: list[tuple[Optional[str], int, Path, str]] = []
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            meta = _read_series_meta(series_dir)
            if meta is None:
                continue
            desc, ser_num = meta
            phase = _classify_phase(desc) or _classify_phase(series_dir.name)
            is_dce = bool(_DCE_RE.search(desc) or _DCE_RE.search(series_dir.name))
            all_series.append((phase, ser_num, series_dir, desc, is_dce))  # type: ignore[arg-type]

        # Açık faz etiketi olan veya DCE filtresiyle eşleşen seriler
        dce_series = [
            (p, n, d, desc)
            for p, n, d, desc, is_dce in all_series
            if p is not None or is_dce
        ]

        if not dce_series:
            continue

        labeled   = [(p, n, d, desc) for p, n, d, desc in dce_series if p]
        unlabeled = sorted(
            [(p, n, d, desc) for p, n, d, desc in dce_series if not p],
            key=lambda x: x[1],
        )

        if not labeled and len(unlabeled) >= 3:
            unlabeled[0]  = ("pre",   *unlabeled[0][1:])
            unlabeled[1]  = ("early", *unlabeled[1][1:])
            unlabeled[-1] = ("late",  *unlabeled[-1][1:])
            labeled = list(unlabeled)

        ps = PhaseSet()
        for phase, _, series_dir, _ in labeled:
            if phase == "pre"   and ps.pre   is None: ps.pre   = series_dir
            elif phase == "early" and ps.early is None: ps.early = series_dir
            elif phase == "late"  and ps.late  is None: ps.late  = series_dir

        # Aynı çalışmadaki etiketlenen fazların seri numarasının altındaki
        # ilk seri → pre-contrast (ISPY1: seri 3 < PE1 seri 31001 gibi)
        # PE1/SER açıklamasının taban adıyla başlayan seri tercih edilir.
        if ps.pre is None and (ps.early or ps.late):
            labeled_nums = [n for p, n, d, desc in labeled if p in ("early", "late")]
            if labeled_nums:
                threshold = min(labeled_nums)

                # PE1/SER'den faz sonekini sil → taban adı
                # Ayırıcı boşluk, alt çizgi, tire veya iki nokta olabilir
                def _strip_phase_suffix(s: str) -> str:
                    return re.sub(
                        r"[\s_:/-]*(pe\s*\d+|ser)\s*$", "", s, flags=re.I
                    ).strip()

                base_names = {
                    _strip_phase_suffix(desc)
                    for p, n, d, desc in labeled
                    if p in ("early", "late")
                }
                base_names.discard("")

                pre_candidates = sorted(
                    [
                        (n, d, desc)
                        for p, n, d, desc, _ in all_series
                        if n < threshold
                    ],
                    key=lambda x: (
                        # Taban adıyla başlayan seri önce gelsin
                        0 if base_names and any(
                            x[2].startswith(b) for b in base_names
                        ) else 1,
                        x[0],  # sonra seri numarasına göre
                    ),
                )
                if pre_candidates:
                    ps.pre = pre_candidates[0][1]

        # pre+early var ama late yok → late = early
        if ps.pre is not None and ps.early is not None and ps.late is None:
            ps.late = ps.early

        # Bu çalışma için en iyi PhaseSet'i güncelle
        if ps.is_complete():
            return ps   # mükemmel eşleşme bulundu
        if best is None or len([x for x in [ps.pre, ps.early, ps.late] if x]) > \
                           len([x for x in [best.pre, best.early, best.late] if x]):
            best = ps

    return best if best is not None else PhaseSet()


# ── 6. Tek hasta işleme hattı ─────────────────────────────────────────────────

def process_patient(patient_dir: Path,
                    out_dir: Path = PROCESSED_DIR,
                    patch_size: int = PATCH_SIZE,
                    patient_id: Optional[str] = None) -> PatientMeta:
    """
    Bir hastanın DCE serilerini yükler, harmonize eder ve kaydeder.

    Çıktı dosyaları:
      <out_dir>/<patient_id>.npy          — (3, 128, 128) float32
      <out_dir>/<patient_id>_meta.json    — işleme metadata

    Returns
    -------
    PatientMeta
    """
    if patient_id is None:
        patient_id = patient_dir.name
    log.info("[%s] Isleniyor ...", patient_id)

    # --- Faz keşfi ---
    phase_set = discover_dce_phases(patient_dir)
    if not phase_set.is_complete():
        raise ValueError(
            f"Eksik DCE fazlari: {phase_set.missing()}  "
            f"(pre={phase_set.pre}, early={phase_set.early}, late={phase_set.late})"
        )

    # --- Yükle ve orientasyonu düzelt ---
    def load_and_orient(series_dir: Path):
        img = _load_sitk_series(series_dir)
        orient_before = get_orientation_code(img)
        img = reorient_to_axial_lps(img)
        orient_after  = get_orientation_code(img)
        return img, orient_before, orient_after

    pre_img,   ob_pre,   oa_pre   = load_and_orient(phase_set.pre)
    early_img, _,        _        = load_and_orient(phase_set.early)
    late_img,  _,        _        = load_and_orient(phase_set.late)

    orig_spacing = tuple(float(s) for s in pre_img.GetSpacing())

    # --- Alan gücü ---
    field_strength = _get_field_strength(phase_set.pre)

    # --- Resample ---
    pre_img   = resample_to_spacing(pre_img)
    early_img = resample_to_spacing(early_img)
    late_img  = resample_to_spacing(late_img)

    pre_vol   = sitk.GetArrayFromImage(pre_img).astype(np.float32)   # (Z,H,W)
    early_vol = sitk.GetArrayFromImage(early_img).astype(np.float32)
    late_vol  = sitk.GetArrayFromImage(late_img).astype(np.float32)

    # Boyutları eşitle (resampling hafif fark bırakabilir)
    target_shape = pre_vol.shape
    early_vol = _match_shape(early_vol, target_shape)
    late_vol  = _match_shape(late_vol,  target_shape)

    # --- Doku maskesi (Otsu) ---
    mask = _tissue_mask(pre_vol)

    # --- Normalizasyon ---
    pre_vol   = normalize_volume(pre_vol,   field_strength, mask)
    early_vol = normalize_volume(early_vol, field_strength, mask)
    late_vol  = normalize_volume(late_vol,  field_strength, mask)

    # --- SER → tümör merkezi ---
    ser    = compute_ser(pre_vol, early_vol, late_vol)
    center = find_tumor_center(ser)

    # --- Patch oluştur ---
    rgb_patch = build_rgb_patch(pre_vol, early_vol, late_vol,
                                center, patch_size)   # (3, H, W)

    # --- Kaydet ---
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{patient_id}.npy", rgb_patch)

    meta = PatientMeta(
        patient_id=patient_id,
        field_strength=field_strength,
        original_spacing=orig_spacing,
        resampled_spacing=TARGET_SPACING,
        orientation_before=ob_pre,
        orientation_after=oa_pre,
        tumor_center_voxel=center,
        patch_size=patch_size,
        phases_used={
            "pre":   str(phase_set.pre),
            "early": str(phase_set.early),
            "late":  str(phase_set.late),
        },
        n_slices_pre=int(pre_vol.shape[0]),
        status="ok",
    )

    with open(out_dir / f"{patient_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

    log.info(
        "[%s] Tamamlandi. Patch: %s  Merkez: %s  Alan: %.1fT",
        patient_id, rgb_patch.shape, center, field_strength,
    )
    return meta


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _tissue_mask(volume: np.ndarray) -> np.ndarray:
    """
    Otsu eşiğiyle doku maskesi oluşturur (arka plan/hava = 0).
    """
    p5 = np.percentile(volume, 5)
    p95 = np.percentile(volume, 95)
    clipped = np.clip(volume, p5, p95)
    norm = ((clipped - p5) / (p95 - p5 + 1e-6) * 255).astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(norm)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = sitk.GetArrayFromImage(otsu.Execute(sitk_img))
    return mask.astype(np.uint8)


def _match_shape(volume: np.ndarray,
                 target: tuple[int, int, int]) -> np.ndarray:
    """
    Resampling sonrası oluşan ±1 voxel farkını kırpma/pad ile giderir.
    """
    result = volume
    for axis in range(3):
        diff = result.shape[axis] - target[axis]
        if diff > 0:
            # Kırp
            slices = [slice(None)] * 3
            slices[axis] = slice(0, target[axis])
            result = result[tuple(slices)]
        elif diff < 0:
            # Pad
            pad = [(0, 0)] * 3
            pad[axis] = (0, -diff)
            result = np.pad(result, pad, mode="edge")
    return result


# ── 7. Toplu işlem ────────────────────────────────────────────────────────────

_ALL_RAW_DIRS = [RAW_DIR / "ISPY1", RAW_DIR / "QIN-02", RAW_DIR / "NACT-Pilot"]

_PATIENT_DIR_PAT = re.compile(
    r"^(ISPY1_(\d+)|QIN-BREAST-\d+-0*(\d+)|UCSF-BR-0*(\d+))$",
    re.IGNORECASE,
)


def _normalize_patient_id(dirname: str) -> str:
    """Dizin adından sayısal hasta ID'sini çıkarır (data_loader ile tutarlı)."""
    m = _PATIENT_DIR_PAT.match(dirname)
    if not m:
        return dirname
    numeric = m.group(2) or m.group(3) or m.group(4)
    return numeric.lstrip("0") or "0"


def process_all_patients(
    out_dir:    Path = PROCESSED_DIR,
    patch_size: int  = PATCH_SIZE,
) -> dict:
    """
    ISPY1, QIN-02 ve NACT-Pilot altındaki tüm hastaları işler.

    Returns
    -------
    {"ok": [patient_ids], "failed": {patient_id: error}, "total": int}
    """
    patient_dirs: list[Path] = []
    for raw_dir in _ALL_RAW_DIRS:
        if raw_dir.exists():
            patient_dirs.extend([
                p for p in raw_dir.rglob("*")
                if p.is_dir() and _PATIENT_DIR_PAT.match(p.name)
            ])
    patient_dirs = sorted(set(patient_dirs))
    log.info("Toplam %d hasta isleniyor ...", len(patient_dirs))

    results: dict = {"ok": [], "failed": {}, "total": len(patient_dirs)}

    for patient_dir in patient_dirs:
        pid = _normalize_patient_id(patient_dir.name)

        if (out_dir / f"{pid}.npy").exists():
            log.info("[%s] Zaten islenmis, atlaniyor.", pid)
            results["ok"].append(pid)
            continue

        try:
            process_patient(patient_dir, out_dir, patch_size, patient_id=pid)
            results["ok"].append(pid)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            log.warning("[%s] HATA — %s", pid, msg)
            log.debug(traceback.format_exc())
            results["failed"][pid] = msg

            meta = PatientMeta(
                patient_id=pid, field_strength=0.0,
                original_spacing=(0,), resampled_spacing=(0,),
                orientation_before="", orientation_after="",
                tumor_center_voxel=(0,), patch_size=patch_size,
                phases_used={}, n_slices_pre=0,
                status="failed", error=msg,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{pid}_meta.json", "w", encoding="utf-8") as f:
                json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

    _print_batch_summary(results)
    return results


def _print_batch_summary(results: dict) -> None:
    total  = results["total"]
    n_ok   = len(results["ok"])
    n_fail = len(results["failed"])
    sep = "-" * 45
    print(f"\n{sep}")
    print(f"  TOPLU ISLEM OZETI")
    print(sep)
    print(f"  Toplam hasta  : {total}")
    print(f"  Basarili      : {n_ok}")
    print(f"  Basarisiz     : {n_fail}")
    if results["failed"]:
        print("  Hatali hastalar:")
        for pid, err in results["failed"].items():
            print(f"    {pid:<20} {err[:60]}")
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCE-MRI harmonizasyon hatti")
    parser.add_argument("--patient",    default=None,
                        help="Tek hasta ID (ör. ISPY1_001). Verilmezse hepsi islenir.")
    parser.add_argument("--ispy1-dir",  default=str(ISPY1_DIR))
    parser.add_argument("--out-dir",    default=str(PROCESSED_DIR))
    parser.add_argument("--patch-size", default=PATCH_SIZE, type=int)
    args = parser.parse_args()

    ispy1_dir = Path(args.ispy1_dir)
    out_dir   = Path(args.out_dir)

    if args.patient:
        patient_dir = ispy1_dir / args.patient
        if not patient_dir.exists():
            raise SystemExit(f"Hasta dizini bulunamadi: {patient_dir}")
        process_patient(patient_dir, out_dir, args.patch_size)
    else:
        process_all_patients(ispy1_dir, out_dir, args.patch_size)
