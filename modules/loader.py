"""
modules/loader.py

에너지 사용량 백데이터 엑셀을 읽어서
표준 원시데이터(df_raw)를 만드는 모듈.

app.py에서 기대하는 공개 함수
- load_spec()
- load_energy_files(year_to_file)
- get_org_order(df_raw_all)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd


# =========================
# 경로 / 공통 유틸
# =========================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent  # app.py, schema.json 등이 있는 위치라고 가정


def _load_json(name: str) -> Any:
    """
    프로젝트 루트에 있는 JSON 파일을 읽어온다.
    - schema.json
    - formulas.json
    - rules_template.json
    """
    path = PROJECT_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")
    # UTF-8, CP949 모두 시도 (로컬 저장 인코딩에 따라 다를 수 있음)
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    # 그래도 안 되면 바이너리로 읽어서 cp949로 강제 디코딩 시도
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("cp949", errors="replace"))


# =========================
# 1) 스펙 로딩
# =========================

def load_spec():
    """
    엑셀 원본에서 추출해 둔 메타데이터(스키마/수식/룰 템플릿)를 불러온다.
    analyzer 모듈에서 그대로 사용하도록 (schema, formulas, rules_template)를 튜플로 반환한다.
    """
    schema = _load_json("schema.json")
    formulas = _load_json("formulas.json")
    rules_template = _load_json("rules_template.json")
    return schema, formulas, rules_template


# =========================
# 2) df_raw 표준 스키마 정의
# =========================

DF_RAW_COLUMNS = [
    "연도",            # int
    "진행상태",        # 작성중 / 확정 등
    "사업군",          # 본사 / 의료 / 복지 ...
    "소속기관명",      # 중앙보훈병원 / 부산보훈병원 ...
    "연면적/설비용량", # float
    "시설구분",        # 건물 / 차량 등
    "연료",            # 전기 / 가스(LNG) / 지역난방 / 등유 / LPG ...
    "단위",            # kWh / ㎥ / Gcal ...
    "담당자",          # 이름
    # 월별 사용량
    "1월",
    "2월",
    "3월",
    "4월",
    "5월",
    "6월",
    "7월",
    "8월",
    "9월",
    "10월",
    "11월",
    "12월",
    # 연간 합계 / 온실가스
    "연간사용량",
    "온실가스환산량_tCO2eq",
    "면적당온실가스배출량",
]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """문자/공백/콤마가 섞인 숫자 컬럼을 안전하게 float로 변환."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
        .astype(float)
    )


# =========================
# 3) 개별 파일 → 단일 연도 df_raw 변환
# =========================

def _guess_sheet_name_for_year(xls: pd.ExcelFile, year: int) -> str:
    """
    업로드된 엑셀 파일에서 백데이터 시트명을 추정한다.
    우선순위:
    1) f"(백데이터){year}년"
    2) f"(백데이터){str(year)[-2:]}년"
    3) 이름에 "백데이터"가 포함된 첫 번째 시트
    4) 첫 번째 시트 (최후 fallback)
    """
    candidates = list(xls.sheet_names)

    # 1, 2) 정확/축약 표기 매칭
    exact1 = f"(백데이터){year}년"
    exact2 = f"(백데이터){str(year)[-2:]}년"
    if exact1 in candidates:
        return exact1
    if exact2 in candidates:
        return exact2

    # 3) '백데이터'를 이름에 포함하는 첫 시트
    for name in candidates:
        if "백데이터" in name:
            return name

    # 4) 그래도 없으면 첫 시트
    return candidates[0]


def _load_single_year_from_excel(file_obj, year: int) -> pd.DataFrame:
    """
    (백데이터)YYYY년 시트를 읽어서 표준 df_raw 구조로 변환한다.

    엑셀 시트의 컬럼 구조 (왼쪽에서 오른쪽 순서 기준, 0부터 인덱스):
    0: 진행상태
    1: 사업군
    2: 소속기관명
    3: 연면적/설비용량
    4: 시설구분
    5: 연료
    6: 단위
    7: 담당자
    8~19: 1~12월 사용량
    20: 연간사용량 (연단위)
    21: 온실가스 환산량 (tCO2eq)
    22: 면적당 온실가스 배출량

    ※ 엑셀 헤더는 실제로는 모두 '시설내역', '에너지사용량' 등 중복 이름이지만,
       위치(index) 기준으로 의미를 해석한다.
    """
    # Streamlit UploadedFile 객체이든, 경로이든 모두 지원
    # pd.ExcelFile이 파일 핸들을 한 번 더 감쌀 수 있어서,
    # 바이너리 버퍼로 통일 처리
    if hasattr(file_obj, "read"):
        # Streamlit UploadedFile 처럼 .read()가 있는 경우
        data = file_obj.read()
        # 이후 analyzer 등에서 다시 read 할 수 있도록 포인터 초기화
        try:
            file_obj.seek(0)
        except Exception:
            pass
        buffer = io.BytesIO(data)
    else:
        # 파일 경로 문자열로 들어온 경우
        buffer = file_obj

    xls = pd.ExcelFile(buffer)
    sheet_name = _guess_sheet_name_for_year(xls, year)

    # 헤더 위치를 자동으로 찾기보다, 가장 윗줄부터 읽고
    # 첫 행을 헤더로 사용한 뒤 위치기반으로 해석
    df_raw_sheet = pd.read_excel(
        xls,
        sheet_name=sheet_name,
        header=0,
        dtype=str  # 일단 모두 문자열로 읽어서 나중에 형 변환
    )

    # 불필요한 완전 빈 행 제거
    df_raw_sheet = df_raw_sheet.dropna(how="all")
    if df_raw_sheet.empty:
        return pd.DataFrame(columns=DF_RAW_COLUMNS)

    # 컬럼 수가 23 미만이면 잘못된 파일로 간주하고 빈 DF 반환
    if df_raw_sheet.shape[1] < 23:
        return pd.DataFrame(columns=DF_RAW_COLUMNS)

    # 위치 인덱스 → 의미있는 이름으로 매핑
    meta_cols = df_raw_sheet.iloc[:, 0:8].copy()
    usage_monthly = df_raw_sheet.iloc[:, 8:20].copy()
    summary_cols = df_raw_sheet.iloc[:, 20:23].copy()

    meta_cols.columns = [
        "진행상태",
        "사업군",
        "소속기관명",
        "연면적/설비용량",
        "시설구분",
        "연료",
        "단위",
        "담당자",
    ]
    usage_monthly.columns = [f"{m}월" for m in range(1, 13)]
    summary_cols.columns = ["연간사용량", "온실가스환산량_tCO2eq", "면적당온실가스배출량"]

    df = pd.concat([meta_cols, usage_monthly, summary_cols], axis=1)

    # 합계 / 빈 행 제거
    mask_sum = (
        df["사업군"].astype(str).str.contains("합계", na=False)
        | df["소속기관명"].astype(str).str.contains("합계", na=False)
    )
    mask_empty = df["연간사용량"].astype(str).str.strip().isin(["", "nan", "None"])
    df = df[~(mask_sum | mask_empty)].copy()

    # 숫자형 컬럼 변환
    numeric_cols = (
        ["연면적/설비용량"]
        + [f"{m}월" for m in range(1, 13)]
        + ["연간사용량", "온실가스환산량_tCO2eq", "면적당온실가스배출량"]
    )
    for c in numeric_cols:
        df[c] = _coerce_numeric(df[c])

    # 연도 컬럼 추가 및 정렬
    df.insert(0, "연도", int(year))

    # 최종 컬럼 순서 강제
    df = df.reindex(columns=DF_RAW_COLUMNS)

    return df.reset_index(drop=True)


# =========================
# 4) 여러 연도 파일 로딩
# =========================

def load_energy_files(year_to_file: Dict[int, Any]) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    app.py에서 사용하는 엔트리 포인트.

    Parameters
    ----------
    year_to_file : dict[int, UploadedFile 또는 경로]
        {연도: 업로드된 엑셀 파일} 매핑.

    Returns
    -------
    df_raw_all : pd.DataFrame
        여러 연도 파일을 모두 합친 df_raw.
    year_to_raw : dict[int, pd.DataFrame]
        연도별 개별 df_raw.
    """
    all_dfs: List[pd.DataFrame] = []
    year_to_raw: Dict[int, pd.DataFrame] = {}

    for year, file_obj in sorted(year_to_file.items()):
        if file_obj is None:
            continue
        try:
            df_year = _load_single_year_from_excel(file_obj, year)
        except Exception:
            # 특정 연도 파일 파싱 실패 시, 해당 연도만 건너뛰고 계속 진행
            df_year = pd.DataFrame(columns=DF_RAW_COLUMNS)

        if df_year is None or df_year.empty:
            continue

        year_to_raw[year] = df_year
        all_dfs.append(df_year)

    if all_dfs:
        df_raw_all = pd.concat(all_dfs, ignore_index=True)
    else:
        df_raw_all = pd.DataFrame(columns=DF_RAW_COLUMNS)

    return df_raw_all, year_to_raw


# =========================
# 5) 기관 정렬 정보
# =========================

def get_org_order(df_raw_all: pd.DataFrame):
    """
    화면 필터/테이블에서 사용할 기관 정렬 정보를 생성한다.

    반환값 (app.py에서 기대하는 구조):
    - org_type_to_orgs : dict[str, list[str]]
        {사업군: [소속기관명 리스트]} 형태.
        (예: {"본사": ["본사"], "의료": ["중앙보훈병원", "부산보훈병원", ...], ...})
    - org_type_order : list[str]
        사업군 표시 순서 (에너지 사용량 합계 기준 내림차순).
    - org_to_order : dict[str, int]
        소속기관명 전체에 대한 글로벌 정렬 인덱스 (차트/테이블 공통 정렬에 사용).
    """
    if df_raw_all is None or df_raw_all.empty:
        return {}, [], {}

    df = df_raw_all.copy()

    # 기관별 연간 사용량 합계 (전 연료 합산)
    usage_by_org = (
        df.groupby(["사업군", "소속기관명"], dropna=False)["연간사용량"]
        .sum(min_count=1)
        .reset_index()
    )

    # 사업군 내 정렬 (기관별 사용량 내림차순)
    usage_by_org = usage_by_org.sort_values(
        by=["사업군", "연간사용량", "소속기관명"],
        ascending=[True, False, True],
        na_position="last",
    )

    org_type_to_orgs: Dict[str, List[str]] = {}
    for row in usage_by_org.itertuples(index=False):
        org_type = getattr(row, "사업군")
        org_name = getattr(row, "소속기관명")
        if pd.isna(org_type) or pd.isna(org_name):
            continue
        org_type_to_orgs.setdefault(org_type, []).append(org_name)

    # 사업군별 총 사용량 → 표시 순서
    usage_by_type = (
        usage_by_org.groupby("사업군")["연간사용량"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )
    org_type_order = [t for t in usage_by_type.index if pd.notna(t)]

    # 전체 기관에 대한 글로벌 정렬 인덱스
    org_to_order: Dict[str, int] = {}
    idx = 0
    for org_type in org_type_order:
        for org in org_type_to_orgs.get(org_type, []):
            if org not in org_to_order:
                org_to_order[org] = idx
                idx += 1

    return org_type_to_orgs, org_type_order, org_to_order
