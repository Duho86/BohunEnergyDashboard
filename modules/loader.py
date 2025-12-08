# modules/loader.py

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
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
    """master_energy_spec.json 위치를 여러 후보 경로에서 순차적으로 탐색한다."""
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

    detail = "시도한 경로들:\n" + "\n".join(str(c) for c in candidates)
    _log_error(detail)
    raise FileNotFoundError("사양 파일을 찾지 못했습니다: master_energy_spec.json")


@lru_cache(maxsize=1)
def load_spec(spec_path: Optional[Union[str, Path]] = None) -> dict:
    """master_energy_spec.json을 로드해서 dict로 반환한다."""
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
    """모든 표·필터에서 사용할 표준 기관명 순서를 반환."""
    return ORG_ORDER


# ===========================================================
# 기관별 시설군(의료/복지/기타) 매핑
# ===========================================================

# 시설군 매핑이 정의되지 않은 기관명을 한 번만 경고하기 위한 전역 캐시
WARNED_UNKNOWN_FACILITY_ORGS: set[str] = set()

ORG_FACILITY_GROUP: Dict[str, str] = {
    # 의료시설
    "중앙보훈병원": "의료시설",
    "부산보훈병원": "의료시설",
    "광주보훈병원": "의료시설",
    "대구보훈병원": "의료시설",
    "대전보훈병원": "의료시설",
    "인천보훈병원": "의료시설",
    # 복지시설 (요양원)
    "수원보훈요양원": "복지시설",
    "광주보훈요양원": "복지시설",
    "김해보훈요양원": "복지시설",
    "대구보훈요양원": "복지시설",
    "대전보훈요양원": "복지시설",
    "남양주보훈요양원": "복지시설",
    "원주보훈요양원": "복지시설",
    "전주보훈요양원": "복지시설",
    # 기타시설
    "본사": "기타시설",
    "보훈교육연구원": "기타시설",
    "보훈원": "기타시설",
    "보훈재활체육센터": "기타시설",
    "보훈휴양원": "기타시설",
}


# ===========================================================
# 엑셀 컬럼 정규화/검사 유틸
# ===========================================================

# 소속기관명/기관명 계열 후보
ORG_COL_CANDIDATES = ["소속기관명", "기관명", "소속기관", "소속기구"]
# 연면적/설비용량 계열 후보
AREA_COL_CANDIDATES = ["연면적/설비용량", "연면적", "연면적(㎡)"]
# 연간 사용량 계열 후보
ANNUAL_USAGE_COL_CANDIDATES = ["연단위", "연간사용량", "연간 사용량"]
# 시설구분은 이름이 거의 고정
FACILITY_TYPE_COL = "시설구분"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """헤더 공백/개행 제거 등 최소 정규화."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", "", regex=False)
    )
    return df


def _find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"필수 컬럼을 찾을 수 없습니다. 후보: {list(candidates)}, 현재 컬럼: {list(df.columns)}"
    )


# ===========================================================
# 파일명 → 연도
# ===========================================================


def infer_year_from_filename(name: str) -> Optional[int]:
    """파일명에서 연도(20xx)를 추출한다."""
    m = re.search(r"(20[0-9]{2})", name)
    if not m:
        return None
    y = int(m.group(1))
    if 2000 <= y <= 2100:
        return y
    return None


def discover_local_energy_files() -> Dict[int, Path]:
    """PROJECT_ROOT/data 에서 연도 정보를 가진 엑셀 파일을 찾아 {연도: 경로} 매핑 생성."""
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
    """로컬(data/) + 세션 업로드 파일을 합쳐 {연도: 파일객체} 매핑 반환."""
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


def build_df_raw(df_original: pd.DataFrame, year: int) -> pd.DataFrame:
    """연도별 엑셀 시트를 df_raw 형식으로 변환한다.

    출력 df_raw 컬럼:
      - 연도, year
      - 기관명, org_name
      - 시설구분(의료/복지/기타)
      - 연면적
      - 연단위(연간 사용량)
      - 1월~12월: 월별 에너지 사용량 (엑셀 헤더명을 그대로 사용)
    """

    if df_original is None or df_original.empty:
        raise ValueError(f"{year}년 엑셀 원본에 데이터가 없습니다.")

    # -------------------------------------------------------
    # 1) 헤더 정규화
    # -------------------------------------------------------
    df = _normalize_columns(df_original)

    org_col = _find_first_existing_column(df, ORG_COL_CANDIDATES)
    area_col = _find_first_existing_column(df, AREA_COL_CANDIDATES)
    annual_col = _find_first_existing_column(df, ANNUAL_USAGE_COL_CANDIDATES)

    if FACILITY_TYPE_COL not in df.columns:
        raise ValueError(
            f"필수 컬럼 '{FACILITY_TYPE_COL}'(시설구분)을 찾을 수 없습니다."
        )

    # -------------------------------------------------------
    # 2) 집계 행 제거: '합계', '합 계', '소계', '소 계' 모두 제거
    # -------------------------------------------------------
    org_all = df[org_col].astype(str).str.strip()
    org_norm = org_all.str.replace(r"\s+", "", regex=True)

    summary_keywords = {"합계", "소계"}
    mask_exact = org_norm.isin(summary_keywords)
    mask_regex = org_all.str.contains(r"(합\s*계|소\s*계)", regex=True)
    drop_mask = mask_exact | mask_regex

    if drop_mask.any():
        df = df.loc[~drop_mask].copy()

    # 집계행 제거 후 기관명 다시 재계산
    org_series = df[org_col].astype(str).str.strip()

    # -------------------------------------------------------
    # 3) 1월~12월 월별 사용량 컬럼 탐지 및 숫자화
    # -------------------------------------------------------
    # '1월', '1 월', '1월 사용량' 등 다양한 헤더를 허용
    month_col_map: dict[int, str] = {}
    for c in df.columns:
        m = re.search(r"(\d{1,2})\s*월", str(c))
        if not m:
            continue
        month_num = int(m.group(1))
        if 1 <= month_num <= 12 and month_num not in month_col_map:
            month_col_map[month_num] = c

    missing_months = [m for m in range(1, 13) if m not in month_col_map]
    if missing_months:
        # 엑셀 구조가 깨진 경우를 바로 알 수 있도록 명시적 에러
        found = {m: col for m, col in sorted(month_col_map.items())}
        raise ValueError(
            f"{year}년 엑셀에서 1~12월 사용량 컬럼을 모두 찾지 못했습니다. "
            f"누락 월: {missing_months}, 탐지된 월/컬럼: {found}"
        )

    # 원본 헤더 이름 그대로 사용하면서 숫자로 변환 (NaN은 0으로 처리)
    month_series_map: dict[str, pd.Series] = {}
    for m in range(1, 13):
        col = month_col_map[m]
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        month_series_map[col] = s

    # -------------------------------------------------------
    # 4) 숫자 컬럼 정규화 (연면적, 연단위)
    # -------------------------------------------------------
    area_raw = pd.to_numeric(df[area_col], errors="coerce")
    area = area_raw.groupby(org_series).transform(lambda s: s.fillna(s.max()))
    annual_usage = pd.to_numeric(df[annual_col], errors="coerce")

    # -------------------------------------------------------
    # 5) 시설군 매핑
    # -------------------------------------------------------
    org_unique = set(org_series.unique())

    def _is_summary_name(name: str) -> bool:
        n = re.sub(r"\s+", "", str(name))
        return (n in summary_keywords) or ("합계" in n) or ("소계" in n)

    unknown_orgs = sorted(
        o for o in org_unique
        if (o not in ORG_FACILITY_GROUP) and (not _is_summary_name(o))
    )

    new_unknowns = [
        o for o in unknown_orgs if o not in WARNED_UNKNOWN_FACILITY_ORGS
    ]
    if new_unknowns:
        _log_warning(
            "시설군 매핑이 정의되지 않은 기관이 있습니다. '기타시설'로 처리합니다: "
            + ", ".join(new_unknowns)
        )
        WARNED_UNKNOWN_FACILITY_ORGS.update(new_unknowns)

    facility_group = org_series.map(ORG_FACILITY_GROUP).fillna("기타시설")

    # -------------------------------------------------------
    # 6) df_raw 구성: 기본 컬럼 + 월별 컬럼
    # -------------------------------------------------------
    data: dict[str, object] = {
        "연도": int(year),
        "year": int(year),
        "기관명": org_series,
        "org_name": org_series,
        "시설구분": facility_group,
        "연면적": area,
        "연단위": annual_usage,
    }
    # 월별 컬럼 추가 (키는 엑셀 헤더명, 예: '1월', '2월', …)
    data.update(month_series_map)

    df_raw = pd.DataFrame(data)
    return df_raw



def load_energy_files(
    year_to_file: Mapping[int, object],
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """연도별 엑셀 파일을 모두 읽어 df_raw_year / df_raw_all 생성."""
    year_to_raw: Dict[int, pd.DataFrame] = {}
    df_list: list[pd.DataFrame] = []

    for year in sorted(year_to_file.keys()):
        file_obj = year_to_file[year]
        try:
            # 엑셀 구조: 0행(보라색 제목), 1행이 실제 헤더이므로 header=1 지정
            df_original = pd.read_excel(file_obj, sheet_name=0, header=1)
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
