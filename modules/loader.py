# modules/loader.py

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None  # type: ignore[assignment]


# ===========================================================
# 경로 / 로그 유틸
# ===========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _log_error(msg: str) -> None:
    if st is not None:
        st.error(msg)
    else:
        print(f"[ERROR] {msg}")


def _log_warning(msg: str) -> None:
    if st is not None:
        st.warning(msg)
    else:
        print(f"[WARN] {msg}")


def _find_spec_path(spec_path: Optional[Union[str, Path]] = None) -> Path:
    """
    master_energy_spec.json 위치를 여러 후보 경로에서 순차적으로 탐색한다.
    """
    candidates = []

    if spec_path is not None:
        candidates.append(Path(spec_path))

    cwd = Path.cwd()

    candidates.extend(
        [
            PROJECT_ROOT / "data" / "master_energy_spec.json",
            PROJECT_ROOT / "master_energy_spec.json",
            cwd / "data" / "master_energy_spec.json",
            cwd / "master_energy_spec.json",
        ]
    )

    for p in candidates:
        if p.is_file():
            return p

    msg = "사양 파일을 찾지 못했습니다: master_energy_spec.json"
    detail = "시도한 경로들:\n" + "\n".join(str(c) for c in candidates)
    _log_error(detail)
    raise FileNotFoundError(msg)


# ===========================================================
# spec 로딩
# ===========================================================


@lru_cache(maxsize=1)
def load_spec(spec_path: Optional[Union[str, Path]] = None) -> dict:
    """
    master_energy_spec.json을 로드해서 dict로 반환한다.
    """
    path = _find_spec_path(spec_path)
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    return spec


# ===========================================================
# 기관명 순서 (사용자 지정 19개 기관)
# ===========================================================

ORG_ORDER: Tuple[str, ...] = (
    "본사",
    "중앙보훈병원",
    "부산보훈병원",
    "광주보훈병원",
    "대구보훈병원",
    "대전보훈병원",
    "인천보훈병원",
    "보훈교육연구원",
    "보훈원",
    "수원보훈요양원",
    "광주보훈요양원",
    "김해보훈요양원",
    "대구보훈요양원",
    "대전보훈요양원",
    "남양주보훈요양원",
    "원주보훈요양원",
    "전주보훈요양원",
    "보훈재활체육센터",
    "보훈휴양원",
)


def get_org_order() -> Tuple[str, ...]:
    """
    모든 표·필터에서 사용할 표준 기관명 순서를 반환.
    """
    return ORG_ORDER


# ===========================================================
# 엑셀 → df_raw 변환을 위한 컬럼 정의
# ===========================================================

# 엑셀 기본 필수 컬럼 (기관명은 없음: 연도별 파일에 따라 없을 수 있어서)
RAW_BASE_COLUMNS = [
    "시설구분",
    "연면적",
]

# 엑셀 사용량 컬럼:
#   첫 번째 컬럼:  "에너지사용량"
#   이후       :  "에너지사용량2" ~ "에너지사용량12"
RAW_ENERGY_COLUMNS = ["에너지사용량"] + [
    f"에너지사용량{m}" for m in range(2, 13)
]

RAW_REQUIRED_COLUMNS = RAW_BASE_COLUMNS + RAW_ENERGY_COLUMNS


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    헤더 공백/개행 제거 등 최소 정규화.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", "", regex=False)
    )
    return df


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """
    엑셀 원본에서 필수 컬럼이 모두 존재하는지 확인.
    누락 시 어떤 컬럼이 없는지와 현재 컬럼 리스트를 함께 보여준다.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 누락되었습니다: {missing}; "
            f"현재 컬럼: {list(df.columns)}"
        )


# ===========================================================
# 파일명 → 연도
# ===========================================================


def infer_year_from_filename(name: str) -> Optional[int]:
    """
    파일명에서 연도(20xx)를 추출한다.
    예: '2024년 에너지 사용량관리.xlsx' -> 2024
    """
    import re

    m = re.search(r"(20[0-9]{2})", name)
    if not m:
        return None
    y = int(m.group(1))
    if 2000 <= y <= 2100:
        return y
    return None


def discover_local_energy_files() -> Dict[int, Path]:
    """
    PROJECT_ROOT/data 에서 연도 정보를 가진 엑셀 파일을 찾아 {연도: 경로} 매핑 생성.
    (로컬 개발 편의용)
    """
    data_dir = PROJECT_ROOT / "data"
    mapping: Dict[int, Path] = {}
    if not data_dir.is_dir():
        return mapping

    for p in data_dir.glob("*.xlsx"):
        y = infer_year_from_filename(p.name)
        if y is None:
            continue
        mapping.setdefault(y, p)
    return mapping


def get_year_to_file() -> Dict[int, object]:
    """
    로컬(data/) + 세션 업로드 파일을 합쳐 {연도: 파일객체} 매핑 반환.
    세션 업로드 파일이 존재하면 로컬 파일보다 우선한다.
    """
    local = discover_local_energy_files()
    session_files: Dict[int, object] = (
        st.session_state.get("year_to_file", {}) if st is not None else {}
    )

    merged: Dict[int, object] = {}
    merged.update(local)
    merged.update(session_files)
    return merged


# ===========================================================
# 엑셀 → df_raw 변환
# ===========================================================


def _extract_org_series(df: pd.DataFrame, year: int) -> pd.Series:
    """
    기관명 컬럼이 없는 구버전 엑셀을 위해 다음 순서로 기관명을 추출한다.

    1) '기관명' 컬럼이 있으면 → 그대로 사용
    2) '소속기관', '소속기구', '소속기관명' 중 하나가 있으면 → 사용
    3) 위가 모두 없고 '시설내역' 컬럼이 있으면 → 시설내역을 기관명으로 간주
    4) 그래도 없으면 → '미상(연도 xxxx)' 고정 문자열 사용
    """
    candidates = ["기관명", "소속기관", "소속기구", "소속기관명"]
    for col in candidates:
        if col in df.columns:
            return df[col].astype(str).str.strip()

    if "시설내역" in df.columns:
        return df["시설내역"].astype(str).str.strip()

    # 마지막 fallback: 전부 동일한 라벨
    return pd.Series([f"미상({year}년)"] * len(df), index=df.index, dtype=object)


def build_df_raw(df_original: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    연도별 엑셀 시트를 df_raw 형식으로 변환한다.

    출력 컬럼:
      - 연도 (int)
      - year (int, 동일 값)
      - 기관명 (문자열)
      - org_name (기관명과 동일)
      - 시설구분 (문자열)
      - 연면적 (float)
      - 연단위 (float, 12개 사용량 합계)
      - 개별 월 사용량 컬럼(에너지사용량, 에너지사용량2~12) 그대로 유지
    """
    if df_original is None or df_original.empty:
        raise ValueError(f"{year}년 엑셀 원본에 데이터가 없습니다.")

    df = _normalize_columns(df_original)

    # 필수 컬럼 체크 (기관명은 필수 아님)
    _ensure_columns(df, RAW_REQUIRED_COLUMNS)

    # 기관명 추출 (여러 컬럼 중에서 찾아서 사용)
    org = _extract_org_series(df, year)

    # 시설구분 / 연면적
    facility_type = df["시설구분"].astype(str).str.strip()
    area = pd.to_numeric(df["연면적"], errors="coerce")

    # 월별 사용량 숫자화
    energy_numeric = {}
    for col in RAW_ENERGY_COLUMNS:
        energy_numeric[col] = pd.to_numeric(df[col], errors="coerce")

    energy_df = pd.DataFrame(energy_numeric)

    # 연간 사용량: 월별 합계 (결측치는 0으로 간주)
    annual_usage = energy_df.fillna(0).sum(axis=1)

    # df_raw 구성
    df_raw = pd.DataFrame(
        {
            "연도": int(year),
            "year": int(year),
            "기관명": org,
            "org_name": org,
            "시설구분": facility_type,
            "연면적": area,
            "연단위": annual_usage,
        }
    )

    # 원본 사용량 컬럼도 함께 유지
    for col in RAW_ENERGY_COLUMNS:
        df_raw[col] = energy_df[col]

    return df_raw


def load_energy_files(
    year_to_file: Mapping[int, object],
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    연도별 엑셀 파일을 모두 읽어 df_raw_year / df_raw_all 생성.

    Parameters
    ----------
    year_to_file : {연도: 파일객체 또는 Path}

    Returns
    -------
    year_to_raw : {연도: df_raw_year}
    df_raw_all  : 모든 연도 df_raw 행을 concat 한 DataFrame
    """
    year_to_raw: Dict[int, pd.DataFrame] = {}
    df_list: list[pd.DataFrame] = []

    for year in sorted(year_to_file.keys()):
        file_obj = year_to_file[year]
        try:
            df_original = pd.read_excel(file_obj)
        except Exception as e:
            _log_error(f"{year}년 에너지 사용량 파일을 읽는 중 오류가 발생했습니다: {e}")
            raise

        try:
            df_year = build_df_raw(df_original, year)
        except Exception as e:
            _log_error(f"{year}년 df_raw 생성 중 오류가 발생했습니다: {e}")
            raise

        year_to_raw[year] = df_year
        df_list.append(df_year)

    if df_list:
        df_raw_all = pd.concat(df_list, ignore_index=True)
    else:
        df_raw_all = pd.DataFrame()

    return year_to_raw, df_raw_all
