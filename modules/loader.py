# modules/loader.py

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import pandas as pd

try:
    import streamlit as st
except ImportError:  # noqa: D401
    # Streamlit이 없는 환경(단위 테스트 등)에서도 동작하도록 하기 위한 fallback
    st = None  # type: ignore[assignment]


# =======================================================================
# 1. 경로 / 공통 유틸
# =======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "data" / "master_energy_spec.json"


def _log_error(msg: str) -> None:
    """Streamlit 환경이면 st.error, 아니면 print로 에러 메시지를 출력."""
    if st is not None:
        st.error(msg)
    else:
        print(f"[ERROR] {msg}")


def _log_warning(msg: str) -> None:
    """Streamlit 환경이면 st.warning, 아니면 print로 경고 메시지를 출력."""
    if st is not None:
        st.warning(msg)
    else:
        print(f"[WARN] {msg}")


# =======================================================================
# 2. master_energy_spec.json 로딩
# =======================================================================


@lru_cache(maxsize=1)
def load_spec(spec_path: Optional[Union[str, Path]] = None) -> dict:
    """
    master_energy_spec.json을 로드해서 dict로 반환한다.

    Parameters
    ----------
    spec_path:
        명시적인 경로를 넘기지 않으면 기본값 data/master_energy_spec.json 사용.

    Raises
    ------
    FileNotFoundError:
        spec 파일을 찾지 못한 경우.
    json.JSONDecodeError:
        JSON 파싱에 실패한 경우.
    """
    path = Path(spec_path) if spec_path is not None else DEFAULT_SPEC_PATH

    if not path.is_file():
        msg = f"사양 파일을 찾지 못했습니다: {path.name}"
        _log_error(msg)
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    return spec


# =======================================================================
# 3. 기관명 / 시설구분 표준화
# =======================================================================

# 사용자 요구에 따른 기관 표시 고정 순서 (표/필터 공통)
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

# 짧은 이름 / 과거 엑셀 표기 등을 긴 기관명으로 매핑
_ORG_SYNONYMS: Mapping[str, str] = {
    # 그대로 사용되는 정식 명칭들
    "본사": "본사",
    "중앙보훈병원": "중앙보훈병원",
    "부산보훈병원": "부산보훈병원",
    "광주보훈병원": "광주보훈병원",
    "대구보훈병원": "대구보훈병원",
    "대전보훈병원": "대전보훈병원",
    "인천보훈병원": "인천보훈병원",
    "보훈교육연구원": "보훈교육연구원",
    "보훈원": "보훈원",
    "보훈휴양원": "보훈휴양원",
    "수원보훈요양원": "수원보훈요양원",
    "광주보훈요양원": "광주보훈요양원",
    "김해보훈요양원": "김해보훈요양원",
    "대구보훈요양원": "대구보훈요양원",
    "대전보훈요양원": "대전보훈요양원",
    "남양주보훈요양원": "남양주보훈요양원",
    "원주보훈요양원": "원주보훈요양원",
    "전주보훈요양원": "전주보훈요양원",
    "보훈재활체육센터": "보훈재활체육센터",
    # 과거 data_3/json 등에서 사용된 짧은 명칭들
    "중앙병원": "중앙보훈병원",
    "부산병원": "부산보훈병원",
    "광주병원": "광주보훈병원",
    "대구병원": "대구보훈병원",
    "대전병원": "대전보훈병원",
    "인천병원": "인천보훈병원",
    "교육연구원": "보훈교육연구원",
    "휴양원": "보훈휴양원",
    "수원요양원": "수원보훈요양원",
    "광주요양원": "광주보훈요양원",
    "김해요양원": "김해보훈요양원",
    "대구요양원": "대구보훈요양원",
    "대전요양원": "대전보훈요양원",
    "남양주요양원": "남양주보훈요양원",
    "원주요양원": "원주보훈요양원",
    "전주요양원": "전주보훈요양원",
    "재활체육센터": "보훈재활체육센터",
}

# 시설구분 표준화 매핑
_FACILITY_TYPE_MAP: Mapping[str, str] = {
    # 의료시설
    "의료": "의료시설",
    "병원": "의료시설",
    "의료시설": "의료시설",
    "의료기관": "의료시설",
    "중앙보훈병원": "의료시설",
    "부산보훈병원": "의료시설",
    "광주보훈병원": "의료시설",
    "대구보훈병원": "의료시설",
    "대전보훈병원": "의료시설",
    "인천보훈병원": "의료시설",
    # 복지시설 (요양원/휴양원/재활 포함)
    "복지": "복지시설",
    "요양원": "복지시설",
    "요양시설": "복지시설",
    "복지시설": "복지시설",
    "휴양원": "복지시설",
    "보훈휴양원": "복지시설",
    "재활체육센터": "복지시설",
    "보훈재활체육센터": "복지시설",
    # 기타시설 (본사, 연구원 등)
    "기타": "기타시설",
    "사무": "기타시설",
    "본사": "기타시설",
    "연구": "기타시설",
    "연구원": "기타시설",
    "보훈교육연구원": "기타시설",
    "기타시설": "기타시설",
}


def get_org_order() -> Tuple[str, ...]:
    """표/필터에 사용할 기관 고정 순서를 반환."""
    return ORG_ORDER


def normalize_org_name(raw_name: str) -> str:
    """
    원본 기관명을 JSON 기준 표준 기관명으로 변환한다.

    불명확한 값이 들어오면 ValueError를 발생시켜 조용히 잘못 매핑되는 것을 막는다.
    """
    if raw_name is None:
        raise ValueError("기관명에 None 값이 포함되어 있습니다.")

    name = str(raw_name).strip()

    if not name:
        raise ValueError("기관명에 빈 문자열이 포함되어 있습니다.")

    # 정확히 일치하는 경우
    if name in _ORG_SYNONYMS:
        return _ORG_SYNONYMS[name]

    # 대소문자/공백 차이를 간단히 허용 (주로 영문 표기가 있을 경우 대비)
    key = name.replace(" ", "")
    for k, v in _ORG_SYNONYMS.items():
        if key == k.replace(" ", ""):
            return v

    allowed = ", ".join(ORG_ORDER)
    raise ValueError(f"알 수 없는 기관명입니다: '{name}'. 허용 값: {allowed}")


def normalize_facility_type(raw_type: str) -> str:
    """
    원본 시설구분을 '의료시설/복지시설/기타시설' 3가지로 표준화한다.
    """
    if raw_type is None:
        raise ValueError("시설구분에 None 값이 포함되어 있습니다.")

    text = str(raw_type).strip()

    if not text:
        raise ValueError("시설구분에 빈 문자열이 포함되어 있습니다.")

    # 완전 일치
    if text in _FACILITY_TYPE_MAP:
        return _FACILITY_TYPE_MAP[text]

    # 부분 문자열 기준 단순 매핑 (예: '의료기관', '요양병원' 등)
    lowered = text.lower()
    if "의료" in lowered or "병원" in lowered:
        return "의료시설"
    if "요양" in lowered or "휴양" in lowered or "재활" in lowered or "복지" in lowered:
        return "복지시설"
    if "본사" in lowered or "연구" in lowered:
        return "기타시설"

    # 그래도 매핑이 안 되면 명시적 에러
    raise ValueError(f"알 수 없는 시설구분입니다: '{text}'. "
                     "의료/복지/요양/연구/본사/기타 등을 사용해 주세요.")


# =======================================================================
# 4. 연도별 엑셀 → 표준 df_raw 변환
# =======================================================================

RequiredColumns = Tuple[str, ...]


def _ensure_columns(df: pd.DataFrame, required: RequiredColumns) -> None:
    """필수 컬럼이 모두 존재하는지 검증한다."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing}. "
                         f"현재 컬럼: {list(df.columns)}")


def _coerce_numeric_series(
    s: pd.Series,
    column_name: str,
    allow_na: bool = False,
) -> pd.Series:
    """
    숫자 시리즈로 변환한다.

    - NaN 허용 여부는 allow_na로 제어.
    - NaN을 0으로 조용히 치환하지 않는다.
    """
    coerced = pd.to_numeric(s, errors="coerce")

    if not allow_na and coerced.isna().any():
        na_cnt = int(coerced.isna().sum())
        raise ValueError(
            f"컬럼 '{column_name}'에 숫자로 변환할 수 없는 값 또는 NaN이 {na_cnt}개 있습니다."
        )
    return coerced


def build_df_raw(df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """
    원본 엑셀 DataFrame을 표준 df_raw 스키마로 변환한다.

    Parameters
    ----------
    df:
        업로드한 엑셀을 pandas로 읽어온 원본 DataFrame.
    year:
        엑셀에 '연도' 컬럼이 없는 경우, 이 인자를 사용해 연도를 지정한다.

    Returns
    -------
    pd.DataFrame
        컬럼: ['기관명', '시설구분', '연면적', 'U', 'W', 'V', '연도']
    """
    # 우선 컬럼 이름에서 공백 제거
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 필수 컬럼 존재 여부 확인
    required_base: RequiredColumns = ("기관명", "시설구분", "연면적")
    _ensure_columns(df, required_base)

    # 연도 처리: '연도' 컬럼이 없으면 year 인자 사용
    if "연도" not in df.columns:
        if year is None:
            raise ValueError(
                "엑셀에 '연도' 컬럼이 없으며, build_df_raw(year=...) 인자도 전달되지 않았습니다."
            )
        df["연도"] = year

    # 숫자 컬럼 변환 (NaN 허용 X)
    df["연도"] = _coerce_numeric_series(df["연도"], "연도", allow_na=False).astype(int)
    df["연면적"] = _coerce_numeric_series(df["연면적"], "연면적", allow_na=False)

    # 에너지 컬럼(U/V/W): 컬럼이 없으면 0으로 생성, 있으면 숫자로 변환 (NaN 허용 X)
    for col in ("U", "V", "W"):
        if col not in df.columns:
            _log_warning(f"엑셀에 '{col}' 컬럼이 없어 0으로 채워진 컬럼을 생성합니다.")
            df[col] = 0
        else:
            df[col] = _coerce_numeric_series(df[col], col, allow_na=False).fillna(0)

    # 기관명 / 시설구분 표준화
    df["기관명"] = df["기관명"].map(normalize_org_name)
    df["시설구분"] = df["시설구분"].map(normalize_facility_type)

    # 최종 스키마만 남기기
    df_raw = df[["기관명", "시설구분", "연면적", "U", "W", "V", "연도"]].copy()

    return df_raw


# =======================================================================
# 5. 여러 연도 파일 로딩 헬퍼
# =======================================================================

ExcelLike = Union[str, Path, "pd.ExcelFile", "pd.io.excel._base.ExcelFile"]  # type: ignore[name-defined]
Uploaded = Union[ExcelLike, "st.runtime.uploaded_file_manager.UploadedFile"]  # type: ignore[name-defined]


def _read_excel(file: Uploaded) -> pd.DataFrame:
    """Streamlit UploadedFile 또는 경로/파일 객체를 DataFrame으로 읽는다."""
    if hasattr(file, "read"):  # Streamlit UploadedFile 등 file-like
        return pd.read_excel(file)
    return pd.read_excel(file)  # 경로(str/Path) 등


def load_energy_files(
    year_to_file: Mapping[int, Uploaded],
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    연도별 업로드 파일들을 읽어 표준 df_raw dict와 전체 concat DataFrame을 반환한다.

    Parameters
    ----------
    year_to_file:
        {연도: 업로드 파일} 형태의 매핑.
        - 연도는 int
        - 값은 Streamlit UploadedFile 또는 파일 경로 등

    Returns
    -------
    year_to_raw:
        {연도: df_raw_연도} 매핑
    df_raw_all:
        모든 연도를 concat한 단일 DataFrame
    """
    year_to_raw: Dict[int, pd.DataFrame] = {}

    for year, file in year_to_file.items():
        if file is None:
            continue

        try:
            df_original = _read_excel(file)
        except Exception as e:  # noqa: BLE001
            msg = f"{year}년 에너지 사용량 파일을 읽는 중 오류가 발생했습니다: {e}"
            _log_error(msg)
            raise

        try:
            df_raw_year = build_df_raw(df_original, year=year)
        except Exception as e:  # noqa: BLE001
            msg = f"{year}년 df_raw 생성 중 오류가 발생했습니다: {e}"
            _log_error(msg)
            raise

        year_to_raw[year] = df_raw_year

    if not year_to_raw:
        raise ValueError("로딩된 에너지 사용량 파일이 없습니다. 연도별 파일을 업로드해 주세요.")

    # 모든 연도 concat
    df_raw_all = pd.concat(year_to_raw.values(), ignore_index=True)

    # 기관 순서로 정렬 (분석기에서 groupby 결과가 항상 동일 순서가 되도록)
    category = pd.CategoricalDf = pd.Categorical(  # type: ignore[assignment]
        df_raw_all["기관명"],
        categories=list(ORG_ORDER),
        ordered=True,
    )
    df_raw_all = df_raw_all.assign(기관명=category).sort_values(
        ["연도", "기관명"],
    ).reset_index(drop=True)

    return year_to_raw, df_raw_all
