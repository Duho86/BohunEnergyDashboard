# modules/loader.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 예외 정의
# ============================================================

class EnergyDataError(Exception):
    """에너지 사용량 엑셀 처리 중 발생하는 도메인 예외."""


# ============================================================
# 기본 설정
# ============================================================

ENERGY_FILENAME_YEAR_PATTERN = re.compile(r"(\d{4})")


def ensure_energy_dir(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)


def _extract_year_from_filename(filename: str) -> int:
    """
    파일명에서 연도(YYYY)를 추출.
    예: '2024년 에너지 사용량관리.xlsx' → 2024
    """
    m = ENERGY_FILENAME_YEAR_PATTERN.search(filename)
    if not m:
        raise EnergyDataError(f"파일명에서 연도를 찾을 수 없습니다: {filename}")
    year = int(m.group(1))
    if year < 2000 or year > 2100:
        raise EnergyDataError(f"비정상 연도({year})가 파일명에서 추출되었습니다: {filename}")
    return year


# ============================================================
# 헤더/컬럼 처리 함수
# ============================================================

def _apply_two_row_header(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    원본 엑셀 구조 :
      row0 : 시설내역/에너지사용량 등 그룹 헤더
      row1 : 진행상태 / 사업군 / 소속기관명 / 연면적 / 시설구분 / 연료 / 1월 / 2월 / …

    → row1을 실제 컬럼 헤더로 사용
    """
    if df_raw.empty:
        raise EnergyDataError("엑셀 원본이 비어 있습니다.")

    if len(df_raw) < 2:
        raise EnergyDataError("2개 행(그룹헤더 + 실제헤더)이 존재해야 합니다.")

    header = df_raw.iloc[1].astype(str).str.strip()
    df = df_raw.iloc[2:].copy()
    df.columns = header

    df = df.loc[:, df.columns.notna()]
    df = df.rename(columns=lambda c: str(c).strip())

    return df


def _detect_facility_column(columns) -> Optional[str]:
    """
    소속기관명 / 기관명 / 소속기관 / 시설명 / 건물명 등 탐지
    """
    cols = [str(c).strip() for c in columns]

    # 우선순위 1: 정확히 '소속기관명'
    for c in cols:
        if c == "소속기관명":
            return c

    # 우선순위 2: '소속기관' 또는 '기관명' 문자열 포함
    for c in cols:
        if ("소속기관" in c) or ("기관명" in c):
            return c

    # 우선순위 3: 시설명 계열
    for c in cols:
        if ("시설명" in c) or ("건물명" in c) or ("시설내역" in c):
            return c

    # 최후 수단: 첫 번째 컬럼
    return cols[0] if cols else None


NON_NUMERIC_SENTINELS = {"-", "_", "", " ", "N/A", "NULL", "null", "nan", "NaN"}


def _coerce_numeric_series(s: pd.Series, col_name: str) -> Tuple[pd.Series, int]:
    """
    문자열·단위(kWh, tCO2eq 등) 제거하고 float로 변환.
    """
    s_str = s.astype(str).str.strip()

    sentinel_mask = s_str.isin(NON_NUMERIC_SENTINELS)
    s_clean = s_str.mask(sentinel_mask, pd.NA)

    # 콤마 제거
    s_clean = s_clean.str.replace(",", "", regex=False)

    # 숫자 / 소수점 / 부호만 추출
    s_clean = s_clean.str.replace(r"[^\d\.\-]", "", regex=True)

    numeric = pd.to_numeric(s_clean, errors="coerce").astype("float64")

    failed_mask = numeric.isna() & s_str.notna() & (~sentinel_mask)
    failed = int(failed_mask.sum())

    return numeric, failed


# ============================================================
# 표준 스키마 변환
# ============================================================

def normalize_energy_dataframe(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    '20xx년 에너지 사용량관리.xlsx' → 표준 스키마(연도, 기관명, 월, 온실가스 환산량)
    """
    df = _apply_two_row_header(df_raw)

    facility_col = _detect_facility_column(df.columns)
    if facility_col is None:
        raise EnergyDataError("소속기관명 컬럼을 찾지 못했습니다.")

    # 월 컬럼 찾기 ('1월' ~ '12월')
    month_cols = [
        c for c in df.columns
        if isinstance(c, str) and c.endswith("월") and c[0].isdigit()
    ]
    if not month_cols:
        raise EnergyDataError("월별 에너지 사용량(1월~12월) 컬럼을 찾지 못했습니다.")

    # 온실가스 환산량
    ghg_col = None
    for c in df.columns:
        if "온실가스" in str(c) and "환산" in str(c):
            ghg_col = c
            break
    if ghg_col is None:
        for c in df.columns:
            if "온실가스" in str(c):
                ghg_col = c
                break

    # 숫자 전처리
    for c in month_cols:
        df[c], _ = _coerce_numeric_series(df[c], c)

    if ghg_col is not None:
        df[ghg_col], _ = _coerce_numeric_series(df[ghg_col], ghg_col)

    # 행별 합계
    df["row_energy_sum"] = df[month_cols].sum(axis=1, skipna=True)

    id_vars = [facility_col, "row_energy_sum"]
    if ghg_col:
        id_vars.append(ghg_col)

    melted = df.melt(
        id_vars=id_vars,
        value_vars=month_cols,
        var_name="월",
        value_name="에너지사용량",
    )

    melted["월"] = melted["월"].str.replace("월", "", regex=False)
    melted["월"] = pd.to_numeric(melted["월"], errors="coerce").astype("Int64")

    if ghg_col:
        melted["row_ghg_total"] = melted[ghg_col]
    else:
        melted["row_ghg_total"] = pd.NA

    melted["온실가스 환산량"] = pd.NA

    mask = (
        melted["row_energy_sum"].notna()
        & (melted["row_energy_sum"] > 0)
        & melted["row_ghg_total"].notna()
    )

    melted.loc[mask, "온실가스 환산량"] = (
        melted.loc[mask, "에너지사용량"]
        / melted.loc[mask, "row_energy_sum"]
        * melted.loc[mask, "row_ghg_total"]
    )

    result = pd.DataFrame({
        "연도": year,
        "기관명": melted[facility_col].astype(str).str.strip(),
        "월": melted["월"],
        "온실가스 환산량": melted["온실가스 환산량"],
    })

    result = result.dropna(subset=["월", "기관명"]).reset_index(drop=True)
    return result


# ============================================================
# U/V/W 원본 분석용 로더 (신규 기능)
# ============================================================

def load_energy_raw_for_analysis(path: Path) -> pd.DataFrame:
    """
    '에너지 사용량관리.xlsx' 원본 시트를 원래 구조 그대로 가져오되,
    2행(실제 헤더)을 컬럼명으로 하는 DataFrame을 반환.

    → U/V/W 컬럼 분석에 필수
    """
    if not path.exists():
        raise EnergyDataError(f"파일이 존재하지 않습니다: {path}")

    try:
        df_raw = pd.read_excel(path, sheet_name=0, header=None)
    except Exception as e:
        raise EnergyDataError(f"엑셀 파일 읽기 오류: {e}")

    header = df_raw.iloc[1].astype(str).str.strip()
    df = df_raw.iloc[2:].copy()
    df.columns = header

    return df.reset_index(drop=True)


# ============================================================
# 업로드 처리 / 변환
# ============================================================

def load_energy_xlsx(path: Path) -> Tuple[pd.DataFrame, int]:
    """표준 스키마 로딩"""
    year = _extract_year_from_filename(path.name)

    try:
        df_raw = pd.read_excel(path, sheet_name=0, header=None)
    except Exception as e:
        raise EnergyDataError(f"엑셀 읽기 오류: {e}")

    try:
        df_std = normalize_energy_dataframe(df_raw, year)
    except Exception as e:
        raise EnergyDataError(f"표준 스키마 변환 오류: {e}")

    return df_std, year


def process_uploaded_energy_file(file_obj, original_filename: str, base_dir: Path):
    ensure_energy_dir(base_dir)

    year = _extract_year_from_filename(original_filename)
    save_path = base_dir / original_filename

    try:
        with open(save_path, "wb") as f:
            f.write(file_obj.getbuffer())
    except Exception as e:
        raise EnergyDataError(f"파일 저장 오류: {e}")

    df_std, _ = load_energy_xlsx(save_path)
    return df_std, year, save_path


# ============================================================
# 구조 진단
# ============================================================

def validate_excel_file(path: Path) -> Dict[str, Any]:
    result = {
        "filename": path.name,
        "ok": False,
        "issues": [],
        "warnings": [],
        "detected_facility_col": None,
        "detected_month_cols": [],
        "detected_ghg_col": None,
    }

    if not path.exists():
        result["issues"].append("파일이 존재하지 않습니다.")
        return result

    try:
        df_raw = pd.read_excel(path, sheet_name=0, header=None)
        df = _apply_two_row_header(df_raw)
    except Exception as e:
        result["issues"].append(f"헤더 처리 오류: {e}")
        return result

    facility_col = _detect_facility_column(df.columns)
    month_cols = [
        c for c in df.columns if isinstance(c, str)
        and c.endswith("월") and c[0].isdigit()
    ]

    ghg_col = None
    for c in df.columns:
        if "온실가스" in str(c) and "환산" in str(c):
            ghg_col = c
            break

    if facility_col is None:
        result["issues"].append("소속기관명 컬럼 탐지 실패")
    if not month_cols:
        result["issues"].append("월별 사용량 컬럼 탐지 실패")
    if ghg_col is None:
        result["warnings"].append("온실가스 환산량 컬럼 없음")

    result["detected_facility_col"] = facility_col
    result["detected_month_cols"] = month_cols
    result["detected_ghg_col"] = ghg_col
    result["ok"] = len(result["issues"]) == 0

    return result
