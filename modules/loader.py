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
# 유틸 / 기본 설정
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
# 헤더/컬럼 탐지 & 숫자 전처리
# ============================================================

def _apply_two_row_header(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    원본 엑셀의
      - 1행: 시설내역 / 에너지사용량 등 그룹 헤더
      - 2행: 진행상태 / 사업군 / 소속기관명 / 연면적 / 시설구분 / 연료 / 1월 / 2월 / …

    구조에서 **2행을 실제 컬럼명**으로 사용하는 형태로 변환.

    - df_raw: pd.read_excel(..., header=None) 으로 읽은 원본
    - 반환: 1번(두 번째) 행을 컬럼명으로 올리고, 데이터는 그 다음 행부터 사용하는 DataFrame
    """
    if df_raw.empty:
        raise EnergyDataError("엑셀 원본이 비어 있습니다.")

    # 두 번째 행을 실제 헤더로 사용
    if len(df_raw) < 2:
        raise EnergyDataError("헤더로 사용할 2개 행(그룹헤더+실제헤더)이 존재하지 않습니다.")

    header = df_raw.iloc[1].astype(str).str.strip()
    df = df_raw.iloc[2:].copy()
    df.columns = header

    # NaN 컬럼 제거 및 공백 제거
    df = df.loc[:, df.columns.notna()]
    df = df.rename(columns=lambda c: str(c).strip())

    return df


def _detect_facility_column(columns) -> Optional[str]:
    """
    소속기관명 컬럼을 찾는다.
    우선순위:
    1) 컬럼명이 정확히 '소속기관명'
    2) '소속기관' 또는 '기관명' 문자열이 포함된 컬럼
    3) 그 외 '시설명', '건물명' 등 시설명을 나타내는 컬럼
    4) 그래도 없으면 첫 번째 컬럼 (fallback)
    """
    cols = [str(c).strip() for c in columns]

    # 1) 정확히 '소속기관명'
    for c in cols:
        if c == "소속기관명":
            return c

    # 2) '소속기관' 또는 '기관명'이 들어간 컬럼
    for c in cols:
        if ("소속기관" in c) or ("기관명" in c):
            return c

    # 3) 시설명 계열
    for c in cols:
        if ("시설명" in c) or ("건물명" in c) or ("시설내역" in c):
            return c

    # 4) 최후의 수단: 첫 번째 컬럼
    return cols[0] if cols else None


NON_NUMERIC_SENTINELS = {"-", "_", "", " ", "N/A", "n/a", "NaN", "nan", "NULL", "null"}


def _coerce_numeric_series(s: pd.Series, col_name: str) -> Tuple[pd.Series, int]:
    """
    문자열/공백/대시/단위 등을 모두 숫자로 전처리하여 float 시리즈로 변환.
    - 콤마, 공백 제거
    - '1,234 tCO2eq', '  500kWh ' 등에서도 숫자 부분만 추출
    - 변환 실패한 값 개수를 함께 반환.
    """
    original = s.copy()

    # 문자열로 변환 후 기본 정리
    s_str = s.astype(str).str.strip()

    # 명시적 sentinel → NaN
    sentinel_mask = s_str.isin(NON_NUMERIC_SENTINELS)
    s_clean = s_str.mask(sentinel_mask, pd.NA)

    # 콤마 제거
    s_clean = s_clean.str.replace(",", "", regex=False)

    # 숫자, 부호, 소수점 외 문자 제거 (단위/문자 제거)
    s_clean = s_clean.str.replace(r"[^\d\.\-]", "", regex=True)

    numeric = pd.to_numeric(s_clean, errors="coerce").astype("float64")

    # "실제값은 뭔가 있었는데 숫자로 안 바뀐" 케이스만 실패로 카운트
    failed_mask = numeric.isna() & s_str.notna() & (s_str != "") & (~sentinel_mask)
    failed_count = int(failed_mask.sum())

    return numeric, failed_count


# ============================================================
# 표준 스키마 변환
# ============================================================

def normalize_energy_dataframe(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    '20xx년 에너지 사용량관리.xlsx' 1개 파일을
    표준 스키마 DataFrame(연도, 기관명, 월, 온실가스 환산량)으로 변환.

    - 헤더 2행 구조:
        row0: 시설내역/에너지사용량 등 그룹 헤더
        row1: 진행상태/사업군/소속기관명/연면적/시설구분/연료/1월/2월/…
    """

    # 1) 헤더 정리 (2행을 실제 컬럼명으로 사용)
    df = _apply_two_row_header(df_raw)

    # 2) 기관명 컬럼 찾기
    facility_col = _detect_facility_column(df.columns)
    if facility_col is None:
        raise EnergyDataError("소속기관명 컬럼을 찾지 못했습니다.")

    # 3) 월 컬럼 찾기 ('1월' ~ '12월')
    month_cols: List[str] = [
        c
        for c in df.columns
        if isinstance(c, str)
        and c.endswith("월")
        and c[0].isdigit()
    ]
    if not month_cols:
        raise EnergyDataError("월별 에너지 사용량 컬럼(1월~12월)을 찾을 수 없습니다.")

    # 4) 온실가스 환산량(연간 또는 합계) 컬럼 탐지
    ghg_col: Optional[str] = None
    for c in df.columns:
        sc = str(c)
        if "온실가스" in sc and "환산" in sc:
            ghg_col = c
            break
    if ghg_col is None:
        for c in df.columns:
            sc = str(c)
            if "온실가스" in sc:
                ghg_col = c
                break

    # 5) 숫자 컬럼 전처리 (월별)
    for c in month_cols:
        df[c], _ = _coerce_numeric_series(df[c], c)

    # 6) 온실가스 환산량(연간) 전처리
    if ghg_col is not None:
        df[ghg_col], _ = _coerce_numeric_series(df[ghg_col], ghg_col)

    # 7) 행별 월 사용량 합계 → 비율 계산용
    df["row_energy_sum"] = df[month_cols].sum(axis=1, skipna=True)

    # 8) 월별 long 포맷으로 변환
    id_vars = [facility_col, "row_energy_sum"]
    if ghg_col is not None:
        id_vars.append(ghg_col)

    melted = df.melt(
        id_vars=id_vars,
        value_vars=month_cols,
        var_name="월",
        value_name="에너지사용량",
    )

    # 9) '1월' → 1
    melted["월"] = melted["월"].astype(str).str.replace("월", "", regex=False)
    melted["월"] = pd.to_numeric(melted["월"], errors="coerce").astype("Int64")

    # 10) 월별 온실가스 환산량 배분
    if ghg_col is not None:
        melted["row_ghg_total"] = melted[ghg_col]
    else:
        melted["row_ghg_total"] = pd.NA

    melted["온실가스 환산량"] = pd.NA

    mask = (
        melted["row_energy_sum"].notna()
        & (melted["row_energy_sum"] > 0)
        & melted["row_ghg_total"].notna()
        & (melted["row_ghg_total"] >= 0)
    )

    melted.loc[mask, "온실가스 환산량"] = (
        melted.loc[mask, "에너지사용량"] / melted.loc[mask, "row_energy_sum"]
        * melted.loc[mask, "row_ghg_total"]
    )

    # 11) 표준 스키마 구성
    result = pd.DataFrame(
        {
            "연도": year,
            "기관명": melted[facility_col].astype(str).str.strip(),
            "월": melted["월"],
            "온실가스 환산량": melted["온실가스 환산량"],
        }
    )

    result = result.dropna(subset=["월", "기관명"])

    return result.reset_index(drop=True)


# ============================================================
# 업로드 파일 처리
# ============================================================

def load_energy_xlsx(path: Path) -> Tuple[pd.DataFrame, int]:
    """
    저장된 '20xx년 에너지 사용량관리.xlsx' 파일을 읽어
    (표준 스키마 DataFrame, 연도) 를 반환.
    """
    if not path.exists():
        raise EnergyDataError(f"파일이 존재하지 않습니다: {path}")

    year = _extract_year_from_filename(path.name)

    try:
        df_raw = pd.read_excel(path, sheet_name=0, header=None)
    except Exception as e:
        raise EnergyDataError(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")

    try:
        df_std = normalize_energy_dataframe(df_raw, year)
    except EnergyDataError:
        raise
    except Exception as e:
        raise EnergyDataError(f"엑셀 데이터를 표준 스키마로 변환하는 중 오류: {e}")

    return df_std, year


def process_uploaded_energy_file(
    file_obj,
    original_filename: str,
    base_dir: Path,
) -> Tuple[pd.DataFrame, int, Path]:
    """
    Streamlit file_uploader 로 업로드된 파일을 data/energy/ 에 저장하고,
    구조를 검증한 뒤 표준 스키마 DataFrame을 반환.
    """
    ensure_energy_dir(base_dir)

    year = _extract_year_from_filename(original_filename)
    save_path = base_dir / original_filename

    try:
        with open(save_path, "wb") as f:
            f.write(file_obj.getbuffer())
    except Exception as e:
        raise EnergyDataError(f"업로드 파일을 저장하는 중 오류가 발생했습니다: {e}")

    df_std, _ = load_energy_xlsx(save_path)

    return df_std, year, save_path


# ============================================================
# 구조 진단 (디버그/테스트용)
# ============================================================

def validate_excel_file(path: Path) -> Dict[str, Any]:
    """
    엑셀 1개 파일 구조 진단.
    - 어떤 컬럼을 기관명으로 인식했는지
    - 월 컬럼은 몇 개 인식했는지
    - 온실가스 환산량 컬럼은 무엇인지
    """
    result: Dict[str, Any] = {
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
    except Exception as e:
        result["issues"].append(f"엑셀 읽기 오류: {e}")
        return result

    try:
        df = _apply_two_row_header(df_raw)
    except Exception as e:
        result["issues"].append(f"헤더 처리 오류: {e}")
        return result

    facility_col = _detect_facility_column(df.columns)
    month_cols = [
        c
        for c in df.columns
        if isinstance(c, str) and c.endswith("월") and c[0].isdigit()
    ]
    ghg_col = None
    for c in df.columns:
        sc = str(c)
        if "온실가스" in sc and "환산" in sc:
            ghg_col = c
            break

    if facility_col is None:
        result["issues"].append("소속기관명 컬럼을 찾을 수 없습니다.")
    if not month_cols:
        result["issues"].append("1월~12월 월별 에너지사용량 컬럼을 찾을 수 없습니다.")
    if ghg_col is None:
        result["warnings"].append("온실가스 환산량(tCO2eq) 컬럼을 찾지 못했습니다.")

    result["detected_facility_col"] = facility_col
    result["detected_month_cols"] = month_cols
    result["detected_ghg_col"] = ghg_col
    result["ok"] = len(result["issues"]) == 0

    return result
