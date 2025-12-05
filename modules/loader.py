import os
import re
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 사양(JSON) 로드 – 필요 시 analyzer 등에서 함께 사용
# ------------------------------------------------------------
SPEC_PATH = "master_energy_spec.json"


@st.cache_data(show_spinner=False)
def load_spec() -> dict:
    """master_energy_spec.json 사양 파일 로드."""
    if not os.path.exists(SPEC_PATH):
        st.error(f"사양 파일을 찾지 못했습니다: {SPEC_PATH}")
        return {}
    try:
        with open(SPEC_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"사양 파일 로딩 실패: {SPEC_PATH} ({e})")
        return {}


# ------------------------------------------------------------
# 내부 유틸 함수들
# ------------------------------------------------------------
def _extract_year_from_filename(filename: str) -> Optional[int]:
    """파일명에서 20xx 형식의 연도를 추출."""
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def _read_raw_excel(path: str) -> pd.DataFrame:
    """공통 원시 엑셀 로더 (header=1)."""
    try:
        df = pd.read_excel(path, sheet_name=0, header=1)
    except Exception as e:
        st.error(f"❌ 파일 로딩 실패: {os.path.basename(path)} ({e})")
        return pd.DataFrame()
    if df is None or df.empty:
        st.error(f"❌ 파일 데이터가 비어 있습니다: {os.path.basename(path)}")
        return pd.DataFrame()
    df.columns = df.columns.map(str)
    return df


def _find_column_by_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    required: bool = True,
    col_label: str = "",
) -> Optional[str]:
    """
    df.columns 중에서 keywords 중 하나 이상을 포함하는 첫 번째 컬럼명을 반환.
    required=True 인데 찾지 못하면 st.error 출력 후 None 반환.
    """
    candidates = [
        c for c in df.columns if any(k in str(c) for k in keywords)
    ]
    if not candidates:
        if required:
            label = col_label or "/".join(keywords)
            st.error(f"❌ '{label}' 컬럼을 찾지 못했습니다.")
        return None
    return candidates[0]


def _extract_month_from_string(val: str) -> Optional[int]:
    """
    '2024-01', '202401', '2024.1', '2024/01' 등 문자열에서 월(1~12)을 추출.
    못 찾으면 None.
    """
    if val is None:
        return None
    s = str(val)
    m = re.search(r"(20\d{2})[-./]?(1[0-2]|0?[1-9])", s)
    if not m:
        return None
    return int(m.group(2))


# ============================================================
# 1) 단일 파일 → df_raw 생성
# ============================================================
def load_energy_raw_for_analysis(path: str, year: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    업로드된 《에너지 사용량관리.xlsx》 파일을 df_raw 형태로 변환하고 반환한다.

    ▷ df_raw 컬럼 정의 (요구사항 반영):
       - 연도      : year 인자 (없으면 None)
       - 기관명    : 소속기관명/기관명/소속기관 중 하나
       - 시설구분  : 시설구분/사업군/용도/구분 중 하나
       - 연면적    : 연면적/면적 계열
       - U        : 연간 에너지 사용량
                    (연단위 컬럼 우선, 없으면 1~12월 합계, 그것도 없으면 사용년월 기반 합계)
       - W        : 평균 에너지 사용량 (월평균)
       - V        : 면적당 온실가스 배출량

    ⚠ 필수 컬럼(기관명/시설구분/연면적/U/V/W) 중 하나라도 제대로 계산되지 않으면
       st.error 로 알리고 None 반환.
    """
    df = _read_raw_excel(path)
    if df.empty:
        return None

    # --------------------------------------------------------
    # 1. 기관명 / 시설구분 / 연면적 컬럼 탐색 (이름 기반)
    # --------------------------------------------------------
    org_col = _find_column_by_keywords(
        df,
        keywords=["소속기관명", "기관명", "소속기관"],
        required=True,
        col_label="소속기관명/기관명",
    )
    fac_col = _find_column_by_keywords(
        df,
        keywords=["시설구분", "사업군", "용도", "구분"],
        required=True,
        col_label="시설구분/사업군",
    )
    area_col = _find_column_by_keywords(
        df,
        keywords=["연면적", "면적"],
        required=True,
        col_label="연면적",
    )
    if org_col is None or fac_col is None or area_col is None:
        # 상세 에러는 각 _find_column_by_keywords 에서 이미 출력됨
        return None

    # --------------------------------------------------------
    # 2. 연간 사용량(U)과 월평균(W) 계산
    # --------------------------------------------------------
    # 2-1) 연단위 사용량 컬럼(연단위 등)
    annual_col = _find_column_by_keywords(
        df,
        keywords=["연단위"],
        required=False,
        col_label="연단위(연간 에너지 사용량)",
    )
    annual_series = None

    # 2-2) 1월~12월 컬럼 탐색
    month_pattern = re.compile(r"^\s*(\d{1,2})월\s*$")
    month_cols = []
    for c in df.columns:
        if month_pattern.match(str(c)):
            month_cols.append(c)

    # 2-3) 사용년월 기반 컬럼(그래프뿐 아니라 U에도 최후 보조용)
    ym_col = None
    for c in df.columns:
        if "사용년월" in str(c):
            ym_col = c
            break

    # ---- U 계산 순서: 연단위 → 1~12월 합계 → 사용년월 합계 ----
    if annual_col is not None:
        df[annual_col] = pd.to_numeric(df[annual_col], errors="coerce")
        if df[annual_col].notna().sum() > 0:
            annual_series = df[annual_col]

    if annual_series is None and month_cols:
        for c in month_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["__U_from_month"] = df[month_cols].sum(axis=1)
        annual_series = df["__U_from_month"]

    if annual_series is None and ym_col is not None:
        # 사용년월 + 에너지 사용량 컬럼 조합으로 연간 합계
        month_series = df[ym_col].astype(str).apply(_extract_month_from_string)
        df["__월"] = month_series
        df = df[df["__월"].notna()]
        if df.empty:
            st.error("❌ '사용년월'에서 월 정보를 추출할 수 있는 행이 없습니다.")
            return None

        # 에너지 사용량 컬럼 탐색
        energy_col = _find_column_by_keywords(
            df,
            keywords=["에너지", "사용"],
            required=False,
            col_label="에너지 사용량",
        )
        if energy_col is None:
            energy_col = _find_column_by_keywords(
                df,
                keywords=["사용량"],
                required=False,
                col_label="사용량",
            )
        if energy_col is None:
            st.error(
                "❌ '사용년월' 기반 연간 집계를 위한 에너지 사용량 컬럼을 찾지 못했습니다."
            )
            return None

        df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
        # 여기서는 행 단위 연간 합계를 만들기 어렵기 때문에,
        # 각 행의 연간 U 는 동일 파일 내 월 합계 기준으로 사용
        # (기관별/행별 분석은 analyzer 쪽에서 집계)
        annual_series = df[energy_col]

    if annual_series is None:
        st.error(
            "❌ 연간 에너지 사용량(U)을 계산할 수 있는 컬럼(연단위, 1~12월, 사용년월)을 찾지 못했습니다."
        )
        return None

    U = pd.to_numeric(annual_series, errors="coerce")

    # ---- W: 월평균 ----
    if month_cols:
        # 월 컬럼 있을 때는 실제 월평균
        for c in month_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        W = df[month_cols].mean(axis=1)
    else:
        # 그 외에는 U/12 를 월평균으로 간주 (명시적 가정)
        W = U / 12.0

    # --------------------------------------------------------
    # 3. V (면적당 온실가스 배출량)
    # --------------------------------------------------------
    v_col = _find_column_by_keywords(
        df,
        keywords=["면적당", "온실가스"],
        required=True,
        col_label="면적당 온실가스 배출량",
    )
    if v_col is None:
        return None
    V = pd.to_numeric(df[v_col], errors="coerce")

    # --------------------------------------------------------
    # 4. df_raw 구성
    # --------------------------------------------------------
    df_raw = pd.DataFrame()
    df_raw["기관명"] = df[org_col].astype(str).str.strip()
    df_raw["시설구분"] = df[fac_col].astype(str).str.strip()
    df_raw["연면적"] = pd.to_numeric(df[area_col], errors="coerce")
    df_raw["U"] = U
    df_raw["W"] = pd.to_numeric(W, errors="coerce")
    df_raw["V"] = V
    df_raw["연도"] = year

    # 필수 숫자 컬럼 검증 (전체 NaN 이면 오류)
    for col in ["U", "W", "연면적", "V"]:
        if df_raw[col].notna().sum() == 0:
            st.error(f"❌ '{col}' 값이 모두 NaN 입니다. 원본 데이터를 확인하세요.")
            return None

    return df_raw


# ============================================================
# 2) 다중 연도 파일 로딩
# ============================================================
def load_all_years(upload_folder: str) -> Tuple[Dict[int, pd.DataFrame], List[str]]:
    """
    업로드 폴더 안의 연도별 파일들을 읽어:
        ({year: df_raw}, [에러메시지]) 반환.

    - 파일명에서 연도 추출 실패 시 해당 파일은 건너뛰고 errors 에 기록
    - df_raw 생성 실패 시 해당 연도는 제외하고 errors 에 기록
    """
    year_to_raw: Dict[int, pd.DataFrame] = {}
    errors: List[str] = []

    if not os.path.exists(upload_folder):
        errors.append("업로드 폴더가 존재하지 않습니다.")
        return {}, errors

    for filename in os.listdir(upload_folder):
        if not filename.lower().endswith(".xlsx"):
            continue

        year = _extract_year_from_filename(filename)
        if year is None:
            errors.append(f"연도를 파일명에서 추출할 수 없습니다: {filename}")
            continue

        path = os.path.join(upload_folder, filename)
        df_raw = load_energy_raw_for_analysis(path, year=year)
        if df_raw is None:
            errors.append(f"{year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda x: x[0]))
    if not year_to_raw:
        errors.append("유효한 연도 파일을 하나도 불러오지 못했습니다.")

    return year_to_raw, errors


# ============================================================
# 3) 월별 에너지 사용량 집계 (대시보드 그래프용)
# ============================================================
def load_monthly_usage(
    upload_folder: str,
    target_year: int,
    selected_orgs: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    특정 연도 파일에서 월별 에너지 사용량 집계.

    ▷ 처리 규칙 (요구사항 2-1 반영):
       1) 1월~12월 컬럼이 있으면: 각 월 합계를 사용
       2) 없고 '사용년월' 컬럼이 있으면: YYYY-MM / YYYYMM 등에서 월을 추출해 집계
       3) 둘 다 없으면: st.error 후 None 반환

    ▷ 반환 형식:
       index = 월(1~12), columns = ['에너지사용량']
    """

    # 1. 해당 연도 파일 찾기
    year_str = str(target_year)
    target_path = None
    for filename in os.listdir(upload_folder):
        if not filename.lower().endswith(".xlsx"):
            continue
        if year_str in filename:
            target_path = os.path.join(upload_folder, filename)
            break

    if target_path is None:
        st.error(f"❌ {target_year}년 파일을 업로드 폴더에서 찾지 못했습니다.")
        return None

    df = _read_raw_excel(target_path)
    if df.empty:
        return None

    # 2. 선택된 소속기구 필터 적용
    if selected_orgs:
        org_col = _find_column_by_keywords(
            df,
            keywords=["소속기관명", "기관명", "소속기관"],
            required=False,
            col_label="소속기관명/기관명",
        )
        if org_col:
            df = df[df[org_col].astype(str).str.strip().isin(selected_orgs)]

    # 3. 우선순위 1: 1월~12월 컬럼
    month_pattern = re.compile(r"^\s*(\d{1,2})월\s*$")
    month_cols = []
    for c in df.columns:
        if month_pattern.match(str(c)):
            month_cols.append(c)

    if month_cols:
        for c in month_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        monthly = df[month_cols].sum(axis=0)

        # 인덱스를 월(1~12) 정수로 교체
        month_index = []
        for c in monthly.index:
            m = month_pattern.match(str(c))
            month_index.append(int(m.group(1)) if m else None)

        monthly.index = month_index
        monthly = monthly.sort_index()
        return pd.DataFrame({"에너지사용량": monthly})

    # 4. 우선순위 2: '사용년월' 컬럼
    ym_col = _find_column_by_keywords(
        df,
        keywords=["사용년월"],
        required=False,
        col_label="사용년월",
    )
    if ym_col is None:
        st.error(
            "❌ 월별 에너지 사용량을 계산할 수 있는 컬럼(1월~12월 또는 '사용년월')을 "
            "원본 파일에서 찾지 못했습니다."
        )
        return None

    df["__사용년월"] = df[ym_col].astype(str)
    df["__월"] = df["__사용년월"].apply(_extract_month_from_string)
    df = df[df["__월"].notna()]

    if df.empty:
        st.error("❌ '사용년월'에서 월 정보를 추출할 수 있는 행이 없습니다.")
        return None

    # 에너지 사용량 컬럼 탐색
    energy_col = _find_column_by_keywords(
        df,
        keywords=["에너지", "사용"],
        required=False,
        col_label="에너지 사용량",
    )
    if energy_col is None:
        energy_col = _find_column_by_keywords(
            df,
            keywords=["사용량"],
            required=False,
            col_label="사용량",
        )

    if energy_col is None:
        st.error(
            "❌ '사용년월' 기반 월별 집계를 위한 에너지 사용량 컬럼을 찾지 못했습니다. "
            "예: '에너지사용량', '전력사용량' 등"
        )
        return None

    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    monthly = df.groupby("__월")[energy_col].sum().sort_index()

    if monthly.empty:
        st.error("❌ '사용년월'과 에너지 사용량을 이용해 월별 합계를 계산할 수 없습니다.")
        return None

    return pd.DataFrame({"에너지사용량": monthly})
