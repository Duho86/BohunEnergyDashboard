import os
import re
import pandas as pd
import streamlit as st


# ============================================================
# 단일 파일 → df_raw 생성
# ============================================================

def load_energy_raw_for_analysis(path: str):
    """
    실제 업로드된 《에너지 사용량관리.xlsx》 파일의 헤더 구조에 맞는 df_raw 생성.
    - 기관명       : 시설내역.2
    - 시설구분     : 시설내역.4
    - 연면적       : 시설내역.3
    - 월별사용량   : 에너지사용량 ~ 에너지사용량.11 (1~12월)
    - U            : 월별 12개월 합계
    - W            : 월별 12개월 평균
    - V            : 면적당 온실가스 배출량
    """
    try:
        df = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        st.error(f"❌ 파일 로딩 실패: {os.path.basename(path)} ({e})")
        return None

    df.columns = df.columns.map(str)

    # --------- 필수 컬럼 존재 여부 확인 ---------
    required_cols = {
        "기관명": "시설내역.2",
        "시설구분": "시설내역.4",
        "연면적": "시설내역.3",
        "V": "면적당 온실가스\n배출량",
    }

    for label, colname in required_cols.items():
        if colname not in df.columns:
            st.error(f"❌ {label}({colname}) 컬럼을 찾지 못했습니다. 파일 헤더를 확인하세요.")
            return None

    # --------- 월별 에너지 사용량 컬럼 탐색 ---------
    month_cols = []
    for c in df.columns:
        if c.startswith("에너지사용량"):
            month_cols.append(c)

    if len(month_cols) != 12:
        st.error(f"❌ 월별 에너지사용량 컬럼이 12개가 아닙니다. 실제={len(month_cols)}")
        return None

    # 월 정렬 (에너지사용량, 에너지사용량.1, ..., 에너지사용량.11)
    month_cols = sorted(month_cols, key=lambda x: (len(x), x))

    # --------- 표준 df_raw 생성 ---------
    df_raw = pd.DataFrame()

    df_raw["기관명"] = df["시설내역.2"]
    df_raw["시설구분"] = df["시설내역.4"]
    df_raw["연면적"] = pd.to_numeric(df["시설내역.3"], errors="coerce")

    # 숫자형 변환
    for c in month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # U = 12개월 합계
    df_raw["U"] = df[month_cols].sum(axis=1)

    # W = 12개월 평균
    df_raw["W"] = df[month_cols].mean(axis=1)

    # V = 면적당 온실가스 배출량
    df_raw["V"] = pd.to_numeric(df["면적당 온실가스\n배출량"], errors="coerce")

    # --------- 숫자형 필수 컬럼 검증 ---------
    for col in ["U", "W", "V", "연면적"]:
        if df_raw[col].notna().sum() == 0:
            st.error(f"❌ '{col}' 값이 모두 NaN 입니다. 원본 데이터를 확인하세요.")
            return None

    return df_raw


# ============================================================
# 다중 연도 로딩
# ============================================================

def _extract_year_from_filename(filename):
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def load_all_years(upload_folder: str):
    year_to_raw = {}
    errors = []

    if not os.path.exists(upload_folder):
        return year_to_raw, errors

    for fn in os.listdir(upload_folder):
        if not fn.endswith(".xlsx"):
            continue

        year = _extract_year_from_filename(fn)
        if year is None:
            errors.append(f"연도 식별 실패: {fn}")
            continue

        path = os.path.join(upload_folder, fn)

        df_raw = load_energy_raw_for_analysis(path)
        if df_raw is None:
            errors.append(f"{year}년 파일 로딩 실패: {fn}")
            continue

        year_to_raw[year] = df_raw

    return dict(sorted(year_to_raw.items())), errors
