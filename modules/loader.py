# modules/loader.py

from __future__ import annotations

import io
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
# 기관명 순서 (사용자 지정 19개 기관 - 소비량/생산량 완벽 동일 적용)
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
    """모든 표·필터·피드백에서 사용할 표준 기관명 순서를 반환."""
    return ORG_ORDER


# ===========================================================
# 에너지원 분류 상수 (요구사항 3: 소비/신재생 생산 필터 분리용)
# ===========================================================

CONSUMPTION_FUELS_ORDER: Tuple[str, ...] = (
    "전기",
    "가스(LNG)",
    "가스(LPG)",
    "등유",
    "지역난방",
)

RENEWABLE_FUELS_ORDER: Tuple[str, ...] = (
    "태양광",
    "지열",
)

RENEWABLE_FUELS = set(RENEWABLE_FUELS_ORDER)
CONSUMPTION_FUELS = set(CONSUMPTION_FUELS_ORDER)


# ===========================================================
# 기관별 시설군(의료/복지/기타) 매핑
# ===========================================================

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

ORG_COL_CANDIDATES = ["소속기관명", "기관명", "소속기관", "소속기구"]
AREA_COL_CANDIDATES = ["연면적/설비용량", "연면적", "연면적(㎡)"]
ANNUAL_USAGE_COL_CANDIDATES = ["연단위", "연간사용량", "연간 사용량"]
FACILITY_TYPE_COL = "시설구분"
ENERGY_TYPE_COL_CANDIDATES = [
    "연료",
    "연료구분",
    "연료 구분",
    "연료종류",
    "연료 종류",
    "에너지종류",
    "에너지 종류",
    "에너지원",
    "에너지원구분",
    "에너지원 구분",
]


def _normalize_energy_type(value: object) -> str:
    """원본의 다양한 연료 표기를 대시보드 표준 에너지 종류로 정규화."""
    if pd.isna(value):
        return "기타"

    raw = str(value).strip()
    compact = re.sub(r"[\s_\-/()]", "", raw).lower()

    if raw in ("", "nan"):
        return "기타"

    if "태양광" in compact or "solar" in compact or "pv" in compact:
        return "태양광"

    if "지열" in compact or "geothermal" in compact:
        return "지열"

    if "전기" in compact or compact in ("electric", "electricity"):
        return "전기"

    if "lpg" in compact or "액화석유가스" in compact:
        return "가스(LPG)"

    if "lng" in compact or "도시가스" in compact or "천연가스" in compact:
        return "가스(LNG)"

    if "등유" in compact or "kerosene" in compact:
        return "등유"

    if (
        "지역난방" in compact
        or "districtheating" in compact
        or "열사용" in compact
        or "열에너지" in compact
    ):
        return "지역난방"

    return raw


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
    """연도별 엑셀 시트를 df_raw 형식으로 변환한다."""
    if df_original is None or df_original.empty:
        raise ValueError(f"{year}년 엑셀 원본에 데이터가 없습니다.")

    df = _normalize_columns(df_original)

    org_col = _find_first_existing_column(df, ORG_COL_CANDIDATES)
    area_col = _find_first_existing_column(df, AREA_COL_CANDIDATES)
    annual_col = _find_first_existing_column(df, ANNUAL_USAGE_COL_CANDIDATES)

    if FACILITY_TYPE_COL not in df.columns:
        raise ValueError(
            f"필수 컬럼 '{FACILITY_TYPE_COL}'(시설구분)을 찾을 수 없습니다."
        )

    energy_type_col = next(
        (c for c in ENERGY_TYPE_COL_CANDIDATES if c in df.columns),
        None,
    )
    if energy_type_col is None:
        energy_type = pd.Series("기타", index=df.index, dtype="object")
        _log_warning(
            f"{year}년 파일에서 연료/에너지 종류 컬럼을 찾지 못했습니다. "
            f"후보 컬럼: {ENERGY_TYPE_COL_CANDIDATES}. 해당 행은 '기타'로 처리합니다."
        )
    else:
        energy_type = df[energy_type_col].map(_normalize_energy_type)

    # 집계 행 제거: '합계', '합 계', '소계', '소 계' 모두 제거
    org_all = df[org_col].astype(str).str.strip()
    org_norm = org_all.str.replace(r"\s+", "", regex=True)

    summary_keywords = {"합계", "소계"}
    mask_exact = org_norm.isin(summary_keywords)
    mask_regex = org_all.str.contains(r"(합\s*계|소\s*계)", regex=True)
    drop_mask = mask_exact | mask_regex

    if drop_mask.any():
        df = df.loc[~drop_mask].copy()

    org_series = df[org_col].astype(str).str.strip()
    if energy_type_col is None:
        energy_type = pd.Series("기타", index=df.index, dtype="object")
    else:
        energy_type = df[energy_type_col].map(_normalize_energy_type)

    # 1월~12월 월별 사용량 컬럼 탐지 및 숫자화
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
        found = {m: col for m, col in sorted(month_col_map.items())}
        raise ValueError(
            f"{year}년 엑셀에서 1~12월 사용량 컬럼을 모두 찾지 못했습니다. "
            f"누락 월: {missing_months}, 탐지된 월/컬럼: {found}"
        )

    month_series_map: dict[str, pd.Series] = {}
    for m in range(1, 13):
        col = month_col_map[m]
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        month_series_map[col] = s

    # 숫자 컬럼 정규화 (연면적, 연단위)
    # 신재생 설비 행에도 기관의 연면적이 동일하게 매핑되도록 기관별 최대값 적용 (0 또는 NaN 방지)
    area_raw = pd.to_numeric(df[area_col], errors="coerce")
    area = area_raw.groupby(org_series).transform(lambda s: s.fillna(s.max())).fillna(1.0)

    annual_usage = pd.to_numeric(df[annual_col], errors="coerce")
    # 연단위가 비어있을 경우 월별 합계로 즉시 대체하여 NaN 원천 방지
    monthly_total_series = pd.DataFrame(month_series_map).sum(axis=1)
    annual_usage = annual_usage.fillna(monthly_total_series).fillna(0.0)

    # 시설군 매핑
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

    data: dict[str, object] = {
        "연도": int(year),
        "year": int(year),
        "기관명": org_series,
        "org_name": org_series,
        "시설구분": facility_group,
        "에너지종류": energy_type,
        "연면적": area,
        "연단위": annual_usage,
    }
    data.update(month_series_map)

    df_raw = pd.DataFrame(data)
    return df_raw


def load_energy_files(
    year_to_file: Mapping[int, object],
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    연도별 엑셀 파일을 모두 읽어 df_raw_year / df_raw_all 생성.
    소비 에너지원과 신재생에너지(태양광, 지열)를 모두 보존하여 로드하고,
    대시보드 선택 상태에 따라 app.py에서 상호 배타적으로 필터링한다. ('용수'만 제외)
    """
    year_to_raw: Dict[int, pd.DataFrame] = {}
    df_list: list[pd.DataFrame] = []

    for year in sorted(year_to_file.keys()):
        file_obj = year_to_file[year]
        try:
            df_original = pd.read_excel(file_obj, sheet_name=0, header=1)
        except Exception as e:
            _log_error(f"{year}년 에너지 사용량 파일을 읽는 중 오류가 발생했습니다: {e}")
            raise

        try:
            normalized = _normalize_columns(df_original)
            fuel_col = next(
                (c for c in ENERGY_TYPE_COL_CANDIDATES if c in normalized.columns), None
            )
            if fuel_col:
                fuel = normalized[fuel_col].astype("string").str.strip()
                normalized = normalized.loc[~fuel.isin(["용수"])].copy()
            df_year = build_df_raw(normalized, year)
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


# ===========================================================
# 하위 호환용 소비 / 생산 분리 데이터 모델
# ===========================================================

def _energy_number(series):
    text = series.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce").astype("float64")


def load_energy_datasets(file_items):
    """[(파일명, bytes)]에서 소비/생산 월별 테이블과 점검 메시지를 만든다."""
    tables, issues, years = [], [], set()
    unit_aliases = {
        "kwh": "kWh", "mwh": "MWh", "gcal": "Gcal",
        "㎥": "m3", "m³": "m3", "m3": "m3", "ℓ": "L", "l": "L",
    }
    for filename, content in file_items:
        year = infer_year_from_filename(filename)
        if year is None:
            raise ValueError(f"{filename}: 파일명에 연도(20xx)가 필요합니다.")
        if year in years:
            raise ValueError(f"{year}년 파일이 중복되었습니다. 연도별 하나만 선택하세요.")
        years.add(year)

        df = pd.read_excel(io.BytesIO(content), sheet_name=0, header=1)
        df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
        if df.columns.duplicated().any():
            raise ValueError(f"{filename}: 중복된 열 이름이 있습니다.")
        months = [f"{m}월" for m in range(1, 13)]
        required = {"소속기관명", "연료", "단위", "연면적/설비용량", *months}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename}: 필수 열 누락 {sorted(missing)}")

        df["원본행"] = df.index + 3
        df["기관명"] = df["소속기관명"].astype("string").str.strip()
        df["에너지원"] = df["연료"].astype("string").str.strip().replace({
            "도시가스": "가스(LNG)", "LNG": "가스(LNG)", "LPG": "가스(LPG)",
        })
        valid = (
            df["기관명"].notna() & df["기관명"].ne("")
            & ~df["기관명"].str.contains(r"합\s*계|소\s*계", na=False)
        )
        df = df.loc[valid].copy()
        supported = CONSUMPTION_FUELS | RENEWABLE_FUELS
        excluded = sorted(set(df["에너지원"].dropna()) - supported)
        if excluded:
            issues.append(f"{year}년 에너지 집계 제외 항목: {', '.join(excluded)}")
        df = df[df["에너지원"].isin(supported)].copy()
        df["구분"] = np.where(df["에너지원"].isin(RENEWABLE_FUELS), "생산", "소비")
        df["연도"] = year
        raw_unit = df["단위"].astype("string").str.strip()
        df["단위"] = raw_unit.str.lower().map(unit_aliases).fillna(raw_unit)

        size = _energy_number(df["연면적/설비용량"])
        df["연면적"] = (
            size.where(df["구분"].eq("소비"))
            .groupby(df["기관명"]).transform("max")
        ).fillna(1.0)
        df["설비용량_원본"] = size.where(df["구분"].eq("생산")).fillna(50.0)
        emission_column = next(
            (c for c in df.columns if c.startswith("온실가스환산량")), None
        )
        df["원본_연간배출량"] = (
            _energy_number(df[emission_column]) if emission_column else 0.0
        )

        for month in months:
            converted = _energy_number(df[month]).fillna(0.0)
            df[month] = converted

        keys = [
            "연도", "원본행", "기관명", "에너지원", "구분", "단위",
            "연면적", "설비용량_원본", "원본_연간배출량",
        ]
        table = df.melt(
            id_vars=keys, value_vars=months, var_name="월", value_name="수량"
        )
        table["월"] = table["월"].str.removesuffix("월").astype(int)
        tables.append(table)

    if not tables:
        raise ValueError("읽을 엑셀 파일이 없습니다.")
    combined = pd.concat(tables, ignore_index=True)
    if combined.empty:
        raise ValueError("분석 가능한 소비 또는 생산 행이 없습니다.")
    return (
        combined[combined["구분"].eq("소비")].copy(),
        combined[combined["구분"].eq("생산")].copy(),
        issues,
    )
