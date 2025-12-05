import json
import os
import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st

SPEC_PATH = "master_energy_spec.json"


@st.cache_data(show_spinner=False)
def load_spec() -> dict:
    if not os.path.exists(SPEC_PATH):
        st.error(f"사양 파일을 찾지 못했습니다: {SPEC_PATH}")
        return {}
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_year_from_filename(filename: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


# ============================================================
# (1) 단일 파일 → df_raw 생성
# ============================================================

def _read_raw_excel(path: str) -> pd.DataFrame:
    """공통 엑셀 로더 (header=1)."""
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


def load_energy_raw_for_analysis(path: str, year: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    업로드된 《에너지 사용량관리.xlsx》 파일을 df_raw 형태로 변환하고 반환한다.

    JSON logic.rules 에서 사용하는 필드명을 기준으로 컬럼을 탐색한다.
      - 연도(year)       : 파일명에서 추출
      - 소속기관명(org)  : '소속기관명', '기관명', '소속기관' 등
      - 연단위(U)        : '연단위'
      - 연면적(area)     : '연면적', '연면적/설비용량' 등
      - 시설구분          : '시설구분', '사업군' 등
      - 면적당 온실가스  : '면적당' + '온실가스'

    반환 df_raw 컬럼:
      ['연도', '기관명', '시설구분', '연면적', 'U', 'V', 'W']
    """
    df = _read_raw_excel(path)
    if df.empty:
        return None

    # -------------------------------
    # 기본 컬럼 탐색
    # -------------------------------
    def pick(col_candidates: List[str], must: bool = True, human_name: str = "") -> Optional[str]:
        cols = [c for c in df.columns if any(k in c for k in col_candidates)]
        if not cols:
            if must:
                st.error(f"❌ '{human_name or '/'.join(col_candidates)}' 컬럼을 찾지 못했습니다.")
            return None
        return cols[0]

    org_col = pick(["소속기관명", "기관명", "소속기관"], human_name="소속기관명/기관명")
    fac_col = pick(["시설구분", "사업군", "용도", "구분"], human_name="시설구분/사업군")
    area_col = pick(["연면적", "면적"], human_name="연면적")
    # 연단위 사용량 (없으면 월합계 사용)
    annual_col = pick(["연단위"], must=False)
    # 면적당 온실가스
    v_col = pick(["면적당", "온실가스"], must=False)

    if org_col is None or fac_col is None or area_col is None:
        return None

    # -------------------------------
    # 월별 사용량 → U/W 생성
    # -------------------------------
    month_cols = [c for c in df.columns if re.fullmatch(r"\d{1,2}월", c)]
    use_month = False
    if annual_col is None or df[annual_col].isna().all():
        # 연단위가 없으면 월합계로
        if month_cols:
            use_month = True
        else:
            # '사용년월' 패턴
            ym_col = None
            for c in df.columns:
                if "사용년월" in c:
                    ym_col = c
                    break
            if ym_col is None:
                st.error("❌ 연단위/월별 사용량/사용년월 컬럼을 찾지 못했습니다.")
                return None
            # 사용년월별 집계는 월별 그래프에서만 사용하므로,
            # 여기서는 연간 합계만 계산
            df["__년"] = df[ym_col].astype(str).str.slice(0, 4)
            df["__U_from_ym"] = pd.to_numeric(df.get("에너지사용량", 0), errors="coerce")
            annual_series = df.groupby([org_col, "__년"])["__U_from_ym"].sum()
    else:
        annual_series = pd.to_numeric(df[annual_col], errors="coerce")

    if use_month:
        for c in month_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["__U_from_month"] = df[month_cols].sum(axis=1)
        annual_series = df["__U_from_month"]

    # W: 월평균 (월 컬럼이 있을 때만)
    if month_cols:
        W = df[month_cols].mean(axis=1)
    else:
        # 월 정보가 없으면 12개월로 균등 가정
        W = annual_series / 12.0

    # V: 면적당 온실가스 배출량
    if v_col:
        V = pd.to_numeric(df[v_col], errors="coerce")
    else:
        V = pd.Series([None] * len(df))

    df_raw = pd.DataFrame()
    df_raw["기관명"] = df[org_col].astype(str).str.strip()
    df_raw["시설구분"] = df[fac_col].astype(str).str.strip()
    df_raw["연면적"] = pd.to_numeric(df[area_col], errors="coerce")
    df_raw["U"] = pd.to_numeric(annual_series, errors="coerce")
    df_raw["W"] = pd.to_numeric(W, errors="coerce")
    df_raw["V"] = V

    # 연도 정보
    if year is None:
        year = None
    df_raw["연도"] = year

    # 숫자형 검증
    for col in ["U", "W", "연면적"]:
        if df_raw[col].notna().sum() == 0:
            st.error(f"❌ '{col}' 값이 모두 NaN 입니다. 원본 파일을 확인하세요.")
            return None

    return df_raw


# ============================================================
# (2) 다중 연도 파일 로딩
# ============================================================

def load_all_years(upload_folder: str) -> Tuple[Dict[int, pd.DataFrame], List[str]]:
    """
    업로드 폴더 안의 연도별 파일들을 읽어:
        ({year: df_raw}, [에러메시지]) 반환.
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
# (3) 월별 사용량 집계 (대시보드 그래프용)
# ============================================================

def load_monthly_usage(
    upload_folder: str, year: int, org_filter: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    특정 연도 파일에서 월별 에너지 사용량 집계.
    - 1월~12월 열이 있으면 합계 사용
    - 없고 '사용년월' 열이 있으면 해당 월을 추출
    반환: index=월(1~12), column='에너지사용량'
    """
    # 해당 연도 파일 찾기
    target_file = None
    for filename in os.listdir(upload_folder):
        if not filename.lower().endswith(".xlsx"):
            continue
        y = _extract_year_from_filename(filename)
        if y == year:
            target_file = os.path.join(upload_folder, filename)
            break

    if target_file is None:
        st.error(f"{year}년 파일을 찾지 못했습니다.")
        return None

    df = _read_raw_excel(target_file)
    if df.empty:
        return None

    # 기관 필터 적용
    if org_filter:
        # 기관 컬럼 탐색
        org_col = None
        for c in df.columns:
            if any(k in c for k in ["소속기관명", "기관명", "소속기관"]):
                org_col = c
                break
        if org_col:
            df = df[df[org_col].astype(str).str.strip().isin(org_filter)]

    # 1월~12월 열 우선
    month_cols = [c for c in df.columns if re.fullmatch(r"\d{1,2}월", c)]
    if month_cols:
        for c in month_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        monthly = df[month_cols].sum(axis=0)
        # '1월' -> 1
        monthly.index = [int(m.replace("월", "")) for m in monthly.index]
        monthly = monthly.sort_index()
        return pd.DataFrame({"에너지사용량": monthly})

    # 사용년월 열
    ym_col = None
    for c in df.columns:
        if "사용년월" in c:
            ym_col = c
            break
    if ym_col is None:
        st.error("월별 사용량을 계산할 수 있는 컬럼(1월~12월, 사용년월)이 없습니다.")
        return None

    df["__사용년월"] = df[ym_col].astype(str)
    df["__월"] = df["__사용년월"].str[-2:].astype(int)
    # 에너지 사용량 컬럼 탐색
    energy_col = None
    for c in df.columns:
        if "에너지" in c and "사용" in c:
            energy_col = c
            break
    if energy_col is None:
        st.error("사용년월 기반 집계를 위한 에너지 사용량 컬럼을 찾지 못했습니다.")
        return None

    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    monthly = df.groupby("__월")[energy_col].sum().sort_index()
    return pd.DataFrame({"에너지사용량": monthly})
