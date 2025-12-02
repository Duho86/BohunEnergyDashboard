# modules/loader.py
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union, List

import pandas as pd


class EnergyDataError(Exception):
    """에너지 사용량 데이터 처리 중 발생하는 공통 예외."""
    pass


# ===========================
# 경로/저장 관련 유틸
# ===========================

def ensure_energy_dir(base_dir: Union[str, Path] = "data/energy") -> Path:
    """
    data/energy 폴더가 없으면 생성하고, Path 객체를 반환한다.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def save_xlsx_file(
    file_obj: BinaryIO,
    original_filename: str,
    base_dir: Union[str, Path] = "data/energy",
) -> Path:
    """
    업로드된 .xlsx 파일을 data/energy/ 폴더에 저장한다.
    """
    energy_dir = ensure_energy_dir(base_dir)

    if not original_filename.lower().endswith(".xlsx"):
        raise EnergyDataError("지원하지 않는 파일 형식입니다. .xlsx 파일만 업로드해 주세요.")

    safe_name = os.path.basename(original_filename)
    dest_path = energy_dir / safe_name

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    try:
        data = file_obj.read()
    except Exception as e:
        raise EnergyDataError(f"업로드된 파일을 읽는 중 오류가 발생했습니다: {e}")

    if not data:
        raise EnergyDataError("업로드된 파일이 비어 있습니다.")

    with open(dest_path, "wb") as out:
        out.write(data)

    return dest_path


# ===========================
# 연도 인식 로직
# ===========================

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def detect_year_from_filename(filename: str) -> Optional[int]:
    match = YEAR_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(0))


def detect_year_from_dataframe(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None

    for col in df.columns:
        if "연도" in str(col):
            year_series = df[col].dropna()
            if not year_series.empty:
                val = str(year_series.iloc[0])
                match = YEAR_PATTERN.search(val)
                if match:
                    return int(match.group(0))

    sample = df.head(10)
    for col in sample.columns:
        for v in sample[col].astype(str):
            match = YEAR_PATTERN.search(v)
            if match:
                return int(match.group(0))

    return None


def detect_year(df: pd.DataFrame, filename: str) -> int:
    year = detect_year_from_filename(filename)
    if year is not None:
        return year

    year = detect_year_from_dataframe(df)
    if year is not None:
        return year

    raise EnergyDataError(
        f"연도를 인식할 수 없습니다. 파일명 또는 시트 내에 연도를 확인해 주세요. (filename={filename})"
    )


# ===========================
# 컬럼 식별/정규화 유틸
# ===========================

def _find_ghg_column(columns: List[str]) -> Optional[str]:
    for col in columns:
        normalized = str(col).replace("\n", "")
        if "온실가스" in normalized and "환산" in normalized:
            return col
    return None


def _find_facility_column(columns: List[str]) -> Optional[str]:
    for col in columns:
        c = str(col)
        if "기관명" in c:
            return col
    for col in columns:
        c = str(col)
        if "시설명" in c:
            return col
    for col in columns:
        c = str(col)
        if "시설내역" in c:
            return col
    return None


def _find_month_columns(columns: List[str]) -> List[str]:
    month_cols: List[str] = []
    for col in columns:
        c = str(col)
        if "에너지" in c and "사용량" in c:
            month_cols.append(col)
    return month_cols


def normalize_energy_dataframe(
    df_raw: pd.DataFrame,
    year: int,
    source_filename: str,
) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        raise EnergyDataError("엑셀 시트에 데이터가 없거나, 모두 비어 있습니다.")

    columns = list(df_raw.columns)

    facility_col = _find_facility_column(columns)
    if facility_col is None:
        raise EnergyDataError(
            "기관/시설을 나타내는 컬럼을 찾을 수 없습니다. (예: '기관명', '시설명', '시설내역')"
        )

    ghg_col = _find_ghg_column(columns)
    if ghg_col is None:
        raise EnergyDataError(
            "['온실가스 환산량'] 컬럼을 찾을 수 없습니다. (예: '온실가스 환산량\\n(tCO2eq)')"
        )

    month_cols = _find_month_columns(columns)
    if not month_cols:
        raise EnergyDataError(
            "월별 에너지 사용량 컬럼을 찾을 수 없습니다. (예: '에너지사용량', '에너지사용량.1' 등)"
        )

    id_vars = [c for c in [facility_col, ghg_col] if c in df_raw.columns]

    try:
        df_melted = df_raw.melt(
            id_vars=id_vars,
            value_vars=month_cols,
            var_name="month_col",
            value_name="에너지사용량",
        )
    except Exception as e:
        raise EnergyDataError(f"월별 데이터 구조를 변환하는 중 오류가 발생했습니다: {e}")

    month_map = {col: idx for idx, col in enumerate(month_cols, start=1)}
    df_melted["월"] = df_melted["month_col"].map(month_map)
    df_melted = df_melted.dropna(subset=["월"])
    df_melted = df_melted.drop(columns=["month_col"])

    df_melted = df_melted.rename(
        columns={
            facility_col: "기관명",
            ghg_col: "온실가스 환산량",
        }
    )

    df_melted["연도"] = int(year)
    df_melted["source_file"] = source_filename

    df_melted = df_melted.dropna(subset=["에너지사용량", "기관명"], how="all")

    if df_melted.empty:
        raise EnergyDataError("정규화 후 남은 유효 데이터가 없습니다. 엑셀 내용을 다시 확인해 주세요.")

    df_std = df_melted[["연도", "기관명", "월", "에너지사용량", "온실가스 환산량", "source_file"]]

    return df_std


# ===========================
# 공개용 상위 함수
# ===========================

def load_energy_xlsx(
    path: Union[str, Path],
) -> Tuple[pd.DataFrame, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    try:
        df_raw = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        raise EnergyDataError(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")

    year = detect_year(df_raw, path.name)
    df_std = normalize_energy_dataframe(df_raw, year=year, source_filename=path.name)

    required_cols = {"기관명", "월", "온실가스 환산량"}
    missing = required_cols - set(df_std.columns)
    if missing:
        raise EnergyDataError(f"필수 컬럼이 누락되었습니다: {missing}")

    return df_std, year


def process_uploaded_energy_file(
    file_obj: BinaryIO,
    original_filename: str,
    base_dir: Union[str, Path] = "data/energy",
) -> Tuple[pd.DataFrame, int, Path]:
    saved_path = save_xlsx_file(file_obj, original_filename, base_dir=base_dir)
    df_std, year = load_energy_xlsx(saved_path)
    return df_std, year, saved_path
