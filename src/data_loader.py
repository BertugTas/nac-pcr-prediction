"""
ISPY1 DICOM veri yükleyici ve klinik tablo birleştirici.

Akış:
  scan_ispy1_patients()  →  DICOM katalog (DataFrame)
  load_clinical()        →  klinik tablo (DataFrame)
  merge_image_clinical() →  birleşik tablo + özet rapor
  load_dicom_volume()    →  tek hasta için 3-D NumPy dizisi
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from config import CLINICAL_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

ISPY1_DIR = RAW_DIR / "ISPY1"

# SeriesDescription'da DCE serisini tanımlamak için aday kelimeler
_DCE_KEYWORDS = re.compile(
    r"dce|dynamic|dyn|post|pre.contrast|contrast|ph\d|phase",
    re.IGNORECASE,
)

# Klinik tablodaki sütun adlarının olası varyantları → standart ada eşleme
_COLUMN_ALIASES: dict[str, list[str]] = {
    "patient_id":  ["patient_id", "patientid", "patient id", "subject_id",
                     "subjectid", "id", "case_id", "nbia id", "nbia_id"],
    "pCR":         ["pcr", "pcr_status", "pcr status", "response",
                     "pathologic_response", "rcb_category"],
    "ER":          ["er", "er_status", "er status", "estrogen", "erpos"],
    "PR":          ["pr", "pr_status", "pr status", "progesterone", "pgrpos", "pgr_pos"],
    "HER2":        ["her2", "her2_status", "her2 status", "her-2", "her2mostpos", "her2pos"],
    "age":         ["age", "age_at_diagnosis", "age at diagnosis"],
    "tumor_size":  ["tumor_size", "tumor size", "size", "longest_dimension",
                     "sld", "ld"],
    "grade":       ["grade", "tumor_grade", "nuclear_grade"],
}

# pCR pozitif olarak kabul edilen değerler
_PCR_POSITIVE = {"1", "1.0", "yes", "y", "true", "pcr", "complete response",
                  "cr", "rcb-0"}


# ── Veri yapıları ─────────────────────────────────────────────────────────────

@dataclass
class SeriesRecord:
    patient_id: str
    study_date: str
    series_uid: str
    series_dir: Path
    series_description: str
    num_files: int
    is_dce: bool
    timepoint: Optional[str] = None   # T0, T1, T2, T3


# ── 1. ISPY1 DICOM tarayıcı ───────────────────────────────────────────────────

def _read_dicom_header(dcm_path: Path) -> Optional[pydicom.Dataset]:
    """İlk DICOM dosyasının başlığını okur; hata varsa None döner."""
    try:
        return pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    except Exception as exc:
        log.debug("DICOM başlık okuma hatası: %s — %s", dcm_path, exc)
        return None


def _is_dce_series(description: str) -> bool:
    return bool(_DCE_KEYWORDS.search(description))


def _assign_timepoints(records: list[SeriesRecord]) -> None:
    """
    Bir hastanın çalışmalarını tarihe göre sıralar ve T0–T3 etiketler.
    En erken tarih T0 (pre-treatment) olarak kabul edilir.
    """
    dates = sorted({r.study_date for r in records if r.study_date})
    date_to_tp = {d: f"T{i}" for i, d in enumerate(dates)}
    for r in records:
        r.timepoint = date_to_tp.get(r.study_date)


def scan_ispy1_patients(
    ispy1_dir: Path = ISPY1_DIR,
    dce_only: bool = True,
    t0_only: bool = True,
) -> pd.DataFrame:
    """
    ISPY1 DICOM klasörünü tarar; her seri için bir satır içeren DataFrame döner.

    Beklenen klasör yapısı (NBIA Data Retriever çıktısı):
        ISPY1/
        └── <PatientID>/
            └── <StudyDate>-<StudyDesc>/
                └── <SeriesNumber>-<SeriesDesc>/
                    └── *.dcm

    Parameters
    ----------
    ispy1_dir : ISPY1 kök dizini
    dce_only  : True → yalnızca DCE serileri
    t0_only   : True → yalnızca T0 (pre-treatment) çalışmaları

    Returns
    -------
    pd.DataFrame — sütunlar: patient_id, study_date, series_uid,
                              series_dir, series_description,
                              num_files, is_dce, timepoint
    """
    if not ispy1_dir.exists():
        log.warning("ISPY1 dizini bulunamadı: %s", ispy1_dir)
        return pd.DataFrame()

    all_records: list[SeriesRecord] = []
    # Manifest dizinleri 1-2 kat derinlikte gerçek hasta klasörlerini içerir.
    # ISPY1_XXXX veya QIN-BREAST-02-XXXX gibi adları olan dizinleri topla.
    _PATIENT_PAT = re.compile(r"^(ISPY1_(\d+)|QIN-BREAST-\d+-(\d+))$")
    patient_dirs = sorted([
        p for p in ispy1_dir.rglob("*")
        if p.is_dir() and _PATIENT_PAT.match(p.name)
    ])
    log.info("Bulunan hasta dizini sayısı: %d", len(patient_dirs))

    for patient_dir in patient_dirs:
        m = _PATIENT_PAT.match(patient_dir.name)
        # Sayısal grup: ISPY1_1001 → "1001", QIN-BREAST-02-0001 → "0001"
        patient_id = (m.group(2) or m.group(3)).lstrip("0") or "0"
        patient_records: list[SeriesRecord] = []

        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue

            # Klasör adından çalışma tarihini çıkar (YYYYMMDD veya MM-DD-YYYY)
            study_date = _extract_date_from_dirname(study_dir.name)

            for series_dir in sorted(study_dir.iterdir()):
                if not series_dir.is_dir():
                    continue

                dcm_files = sorted(series_dir.glob("*.dcm"))
                if not dcm_files:
                    dcm_files = sorted(series_dir.glob("**/*.dcm"))
                if not dcm_files:
                    continue

                header = _read_dicom_header(dcm_files[0])
                if header is None:
                    continue

                series_uid  = getattr(header, "SeriesInstanceUID", series_dir.name)
                series_desc = str(getattr(header, "SeriesDescription", series_dir.name))
                modality    = getattr(header, "Modality", "")

                if modality and modality != "MR":
                    continue

                # Çalışma tarihini DICOM başlığından da almayı dene
                if not study_date:
                    study_date = str(getattr(header, "StudyDate", ""))

                rec = SeriesRecord(
                    patient_id=patient_id,
                    study_date=study_date,
                    series_uid=str(series_uid),
                    series_dir=series_dir,
                    series_description=series_desc,
                    num_files=len(dcm_files),
                    is_dce=_is_dce_series(series_desc),
                )
                patient_records.append(rec)

        if patient_records:
            _assign_timepoints(patient_records)
            all_records.extend(patient_records)

    if not all_records:
        log.warning("Hiç DICOM serisi bulunamadı.")
        return pd.DataFrame()

    df = pd.DataFrame([vars(r) for r in all_records])
    df["series_dir"] = df["series_dir"].astype(str)

    log.info(
        "Toplam %d seri bulundu (%d hasta).",
        len(df), df["patient_id"].nunique(),
    )

    if dce_only:
        df = df[df["is_dce"]].copy()
        log.info("DCE filtresi sonrası: %d seri.", len(df))

    if t0_only:
        df = df[df["timepoint"] == "T0"].copy()
        log.info("T0 filtresi sonrası: %d seri.", len(df))

    # Bir hastada birden fazla DCE T0 serisi varsa en çok dilime sahip olanı tut
    if dce_only and t0_only and not df.empty:
        df = (
            df.sort_values("num_files", ascending=False)
              .drop_duplicates(subset="patient_id", keep="first")
              .reset_index(drop=True)
        )
        log.info("Hasta başına en büyük DCE T0 serisi seçildi: %d hasta.", len(df))

    return df


def _extract_date_from_dirname(name: str) -> str:
    """
    Klasör adından tarihi çıkar.
    Desteklenen formatlar: YYYYMMDD, MM-DD-YYYY, YYYY-MM-DD
    """
    patterns = [
        r"(\d{8})",                        # 20030115
        r"(\d{2})-(\d{2})-(\d{4})",        # 01-15-2003  → YYYYMMDD
        r"(\d{4})-(\d{2})-(\d{2})",        # 2003-01-15  → YYYYMMDD
    ]
    for pat in patterns[1:]:
        m = re.search(pat, name)
        if m:
            parts = m.groups()
            if len(parts) == 3:
                if len(parts[0]) == 4:          # YYYY-MM-DD
                    return "".join(parts)
                else:                            # MM-DD-YYYY
                    return parts[2] + parts[0] + parts[1]
    m = re.search(patterns[0], name)
    return m.group(1) if m else ""


# ── 2. Klinik tablo yükleyici ─────────────────────────────────────────────────

def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Klinik tablodaki sütun adlarını standart isimlere eşler.
    Tanınmayan sütunlar olduğu gibi korunur.
    """
    col_map: dict[str, str] = {}
    normalized = {_normalize_col(c): c for c in df.columns}

    for standard_name, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            norm = _normalize_col(alias)
            if norm in normalized:
                col_map[normalized[norm]] = standard_name
                break

    df = df.rename(columns=col_map)
    return df


def _binarize_pcr(series: pd.Series) -> pd.Series:
    """
    pCR sütununu 0/1 binary'e dönüştürür.
    Tanınmayan değerler NaN olarak işaretlenir ve loglanır.
    """
    def _convert(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in _PCR_POSITIVE:
            return 1
        if s in {"0", "0.0", "no", "n", "false", "non-pcr", "non pcr",
                  "rcb-i", "rcb-ii", "rcb-iii",
                  "pr", "sd", "pd", "partial response",
                  "stable disease", "progressive disease",
                  "npcr", "nomatch", "no match", "non_pcr"}:
            return 0
        return np.nan

    result = series.map(_convert)
    n_unknown = result.isna().sum() - series.isna().sum()
    if n_unknown > 0:
        unknown_vals = series[result.isna() & series.notna()].unique()
        log.warning(
            "pCR sütununda tanınmayan %d değer NaN yapıldı: %s",
            n_unknown, unknown_vals,
        )
    return result.astype("Int64")


_QIN_ID_PAT = re.compile(r"QIN-BREAST-\d+-0*(\d+)", re.IGNORECASE)


def _load_ispy1_clinical(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)

    def _is_data_sheet(name: str) -> bool:
        n = name.lower()
        return "dictionary" not in n and (
            "patient" in n or "subset" in n or "clinical" in n
        )

    sheet = next(
        (s for s in xl.sheet_names if _is_data_sheet(s)),
        xl.sheet_names[0],
    )
    log.info("ISPY1 sheet secildi: '%s'", sheet)
    df = pd.read_excel(xl, sheet_name=sheet, dtype=str)
    df = _map_columns(df)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    outcomes_sheet = next(
        (s for s in xl.sheet_names
         if "outcome" in s.lower() and "dictionary" not in s.lower()),
        None,
    )
    if outcomes_sheet:
        log.info("Outcomes sheet bulundu: '%s'", outcomes_sheet)
        df_out = pd.read_excel(xl, sheet_name=outcomes_sheet, dtype=str)
        df_out = _map_columns(df_out)
        pcr_col = next((c for c in df_out.columns if c.lower() == "pcr"), None)
        if pcr_col and pcr_col != "pCR":
            df_out = df_out.rename(columns={pcr_col: "pCR"})
        if "patient_id" in df_out.columns and "pCR" in df_out.columns:
            df_out["patient_id"] = df_out["patient_id"].astype(str).str.strip()
            df = pd.merge(df, df_out[["patient_id", "pCR"]], on="patient_id", how="left")

    return df


def _load_qin02_clinical(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    df = pd.read_excel(xl, sheet_name=0, dtype=str)
    # İlk satır CDE metadata — sadece gerçek QIN ID içeren satırları al
    df = df[df.iloc[:, 0].str.match(r"QIN-BREAST", na=False)].copy()
    df = _map_columns(df)
    if "patient_id" not in df.columns:
        raise ValueError(f"QIN-02 dosyasında hasta ID sütunu bulunamadı: {path}")
    # QIN-BREAST-02-0001 → "1" (DICOM dizin adıyla aynı format)
    df["patient_id"] = df["patient_id"].map(
        lambda v: (m := _QIN_ID_PAT.search(str(v))) and m.group(1).lstrip("0") or None
    )
    df = df.dropna(subset=["patient_id"])
    log.info("QIN-02 klinik veri yüklendi: %d hasta (%s)", len(df), path.name)
    return df


def load_clinical(
    clinical_dir: Path = CLINICAL_DIR,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Klinik dosyaları yükler (ISPY1 + QIN-02), sütunları standartlaştırır,
    pCR'ı binary yapar ve tek DataFrame döndürür.
    """
    if filename:
        candidates = [clinical_dir / filename]
    else:
        candidates = (
            sorted(clinical_dir.glob("*.xlsx")) +
            sorted(clinical_dir.glob("*.xls")) +
            sorted(clinical_dir.glob("*.csv"))
        )

    if not candidates or not any(p.exists() for p in candidates):
        raise FileNotFoundError(
            f"Klinik dosya bulunamadı: {clinical_dir}\n"
            "Excel (.xlsx) veya CSV dosyasını data/clinical/ altına koyun."
        )

    frames: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        log.info("Klinik dosya yükleniyor: %s", path)
        try:
            if "qin" in path.name.lower():
                df = _load_qin02_clinical(path)
            elif path.suffix in {".xlsx", ".xls"}:
                df = _load_ispy1_clinical(path)
            else:
                df = pd.read_csv(path, dtype=str)
                df = _map_columns(df)
                df["patient_id"] = df["patient_id"].astype(str).str.strip()
            frames.append(df)
        except Exception as exc:
            log.warning("Klinik dosya yüklenemedi (%s): %s", path.name, exc)

    if not frames:
        raise ValueError("Hiçbir klinik dosya yüklenemedi.")

    df = pd.concat(frames, ignore_index=True, sort=False)

    if "patient_id" not in df.columns:
        raise ValueError(
            "Klinik tabloda hasta ID sütunu bulunamadı.\n"
            f"Mevcut sütunlar: {list(df.columns)}"
        )

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset="patient_id", keep="first")

    if "pCR" in df.columns:
        df["pCR"] = _binarize_pcr(df["pCR"])
    else:
        log.warning("pCR sütunu bulunamadı; sonraki adımlar için gereklidir.")

    log.info(
        "Klinik tablo yüklendi: %d hasta, %d sütun.",
        len(df), len(df.columns),
    )
    return df


# ── 3. Görüntü + klinik birleştirici ─────────────────────────────────────────

def merge_image_clinical(
    scan_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    how: str = "inner",
) -> tuple[pd.DataFrame, dict]:
    """
    DICOM katalog ile klinik tabloyu hasta ID üzerinden birleştirir.

    Parameters
    ----------
    scan_df      : scan_ispy1_patients() çıktısı
    clinical_df  : load_clinical() çıktısı
    how          : 'inner' → sadece her iki tabloda da bulunanlar
                   'left'  → tüm DICOM hastaları, eksik klinik NaN

    Returns
    -------
    (merged_df, summary_dict)
    """
    if scan_df.empty:
        log.warning("DICOM kataloğu boş; birleştirme atlanıyor.")
        return pd.DataFrame(), {}

    scan_ids     = set(scan_df["patient_id"])
    clinical_ids = set(clinical_df["patient_id"])

    only_in_scan     = scan_ids - clinical_ids
    only_in_clinical = clinical_ids - scan_ids

    if only_in_scan:
        log.warning(
            "%d hasta DICOM'da var, klinik tabloda yok: %s%s",
            len(only_in_scan),
            ", ".join(sorted(only_in_scan)[:5]),
            " ..." if len(only_in_scan) > 5 else "",
        )
    if only_in_clinical:
        log.info(
            "%d hasta klinik tabloda var, DICOM'da yok (henüz indirilmemiş olabilir).",
            len(only_in_clinical),
        )

    merged = pd.merge(scan_df, clinical_df, on="patient_id", how=how)
    log.info(
        "Birleştirme tamamlandı (%s join): %d eşleşen hasta.",
        how, len(merged),
    )

    summary = _build_summary(merged)
    _print_summary(summary)

    return merged, summary


# ── 4. Özet rapor ─────────────────────────────────────────────────────────────

def _build_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"total_patients": len(df)}

    if "pCR" in df.columns:
        vc = df["pCR"].value_counts(dropna=False)
        summary["pCR_positive"]  = int(vc.get(1, 0))
        summary["pCR_negative"]  = int(vc.get(0, 0))
        summary["pCR_missing"]   = int(df["pCR"].isna().sum())
        total_labeled = summary["pCR_positive"] + summary["pCR_negative"]
        if total_labeled:
            summary["pCR_rate"] = round(
                summary["pCR_positive"] / total_labeled * 100, 1
            )

    standard_cols = ["ER", "PR", "HER2", "age", "tumor_size", "grade"]
    missing: dict[str, int] = {}
    for col in standard_cols:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                missing[col] = n
    summary["missing_values"] = missing

    if "num_files" in df.columns:
        summary["median_slices"] = float(df["num_files"].median())

    return summary


def _print_summary(summary: dict) -> None:
    sep = "─" * 45
    print(f"\n{sep}")
    print("  VERİ ÖZETİ")
    print(sep)
    print(f"  Toplam hasta           : {summary.get('total_patients', '—')}")
    if "pCR_positive" in summary:
        print(f"  pCR pozitif (1)        : {summary['pCR_positive']}")
        print(f"  pCR negatif (0)        : {summary['pCR_negative']}")
        print(f"  pCR etiketsiz          : {summary['pCR_missing']}")
        print(f"  pCR oranı              : %{summary.get('pCR_rate', '—')}")
    if "median_slices" in summary:
        print(f"  Medyan dilim/seri      : {summary['median_slices']:.0f}")
    missing = summary.get("missing_values", {})
    if missing:
        print("  Eksik değerler         :")
        for col, n in missing.items():
            print(f"    {col:<20} {n} hasta")
    else:
        print("  Eksik değerler         : yok")
    print(sep + "\n")


# ── 5. DICOM hacim yükleyici ──────────────────────────────────────────────────

def load_dicom_volume(series_dir: str | Path) -> np.ndarray:
    """
    Bir seri klasöründeki tüm DICOM dilimlerini 3-D NumPy dizisine yükler.

    Returns
    -------
    np.ndarray, şekil (n_slices, height, width), dtype float32
    """
    series_dir = Path(series_dir)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))

    if not dicom_names:
        raise FileNotFoundError(f"Seri dizininde DICOM bulunamadı: {series_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    log.info("Hacim yüklendi: %s  →  %s", series_dir.name, volume.shape)
    return volume


def load_single_dicom(path: str | Path) -> np.ndarray:
    """Tek bir DICOM dosyasını piksel dizisi olarak döner."""
    ds = pydicom.dcmread(str(path))
    return ds.pixel_array.astype(np.float32)


# ── Kullanım örneği ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    scan_df = scan_ispy1_patients(dce_only=True, t0_only=True)
    print(scan_df[["patient_id", "study_date", "series_description",
                    "num_files", "timepoint"]].to_string(index=False))

    clinical_df = load_clinical()

    merged_df, summary = merge_image_clinical(scan_df, clinical_df)

    out = CLINICAL_DIR.parent.parent / "outputs" / "results" / "merged_catalog.csv"
    merged_df.to_csv(out, index=False)
    log.info("Birleşik katalog kaydedildi: %s", out)
