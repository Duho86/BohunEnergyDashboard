import os
import re
import pandas as pd
import streamlit as st


# ============================================================
# (1) 단일 파일 → df_raw 생성
# ============================================================

def load_energy_raw_for_analysis(path: str):
    """
    업로드된 《에너지 사용량관리.xlsx》 파일을 df_raw 형태로 변환하고 반환한다.

    ▣ 이 함수는 다음 조건을 만족해야 한다 (요구사항 0~1절 반영):
        - df_std 사용 금지
        - df_raw 만 사용
        - U/V/W/연면적은 "열 인덱스"가 아닌 "헤더 문자열" 기반 매핑
        - 엑셀 원본 헤더와 동일한 문자열 기반 탐색
        - 찾지 못하면 임의 계산 금지, 명시적 오류
        - 숫자형 변환 필수
        - 월별 1~12월 존재해야 함 (예시 파일 기준)
        - 기관명/시설구분/연면적 컬럼은 헤더 자동 탐색으로 확정

    반환 df_raw 컬럼:
        ['기관명', '시설구분', '연면적', 'U', 'V', 'W']
    """

    # --------------------------
    # ① 파일 읽기
    # --------------------------
    try:
        # 예시 엑셀 구조: 0행은 header title row, 실제 헤더는 row1
        df = pd.read_excel(path, sheet_name=0, header=1)
    except Exception as e:
        st.error(f"❌ 파일 로딩 실패: {os.path.basename(path)} ({e})")
        return None

    if df is None or df.empty:
        st.error(f"❌ 파일 데이터가 비어 있습니다: {os.path.basename(path)}")
        return None

    df.columns = df.columns.map(str)  # 문자열 통일

    # ============================================================
    # (2) 기관명 / 시설구분 / 연면적 자동 탐색
    # ============================================================

    # -- 기관명 후보 (예시 엑셀: '소속기관명')
    org_candidates = [
        c for c in df.columns
        if any(k in c for k in ["소속기관명", "기관명", "소속기관", "부서명", "기관"])
    ]
    if not org_candidates:
        st.error("❌ 기관명(예: '소속기관명') 컬럼을 찾지 못했습니다.")
        return None
    org_col = org_candidates[0]

    # -- 시설구분 후보 (예: '사업군', '시설구분', '용도')
    facility_candidates = [
        c for c in df.columns
        if any(k in c for k in ["사업군", "시설구분", "용도", "구분"])
    ]
    if not facility_candidates:
        st.error("❌ 시설구분(예: '사업군') 컬럼을 찾지 못했습니다.")
        return None
    facility_col = facility_candidates[0]

    # -- 연면적 후보 (예: '연면적', '연면적/설비용량')
    area_candidates = [
        c for c in df.columns
        if ("연면적" in c) or ("면적" in c)
    ]
    if not area_candidates:
        st.error("❌ 연면적(예: '연면적/설비용량') 컬럼을 찾지 못했습니다.")
        return None
    area_col = area_candidates[0]

    # ============================================================
    # (3) 월별 1~12월 컬럼 존재 여부 확인
    # ============================================================

    month_cols = [f"{m}월" for m in range(1, 13)]
    missing = [c for c in month_cols if c not in df.columns]

    if missing:
        st.error(
            f"❌ 월별 에너지 사용량 컬럼(1월~12월) 중 일부가 누락되었습니다.\n"
            f"   누락 컬럼: {missing}"
        )
        return None

    # 월별 숫자형 변환
    for c in month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ============================================================
    # (4) U / W / V 생성
    # ============================================================

    # -- U: 연간 에너지 사용량 (원본 '연단위'가 있으면 그대로 사용)
    if "연단위" in df.columns and df["연단위"].notna().sum() > 0:
        U = pd.to_numeric(df["연단위"], errors="coerce")
    else:
        U = df[month_cols].sum(axis=1)

    # -- W: 평균 에너지 사용량 (월평균)
    W = df[month_cols].mean(axis=1)

    # -- V: 면적당 온실가스 배출량
    v_candidates = [
        c for c in df.columns
        if ("온실가스" in c and "면적" in c)
        or ("면적당" in c and "가스" in c)
    ]
    if not v_candidates:
        st.error("❌ '면적당 온실가스 배출량' 컬럼을 찾지 못했습니다.")
        return None
    V = pd.to_numeric(df[v_candidates[0]], errors="coerce")

    # ============================================================
    # (5) df_raw 최종 생성
    # ============================================================

    df_raw = pd.DataFrame()
    df_raw["기관명"] = df[org_col]
    df_raw["시설구분"] = df[facility_col]
    df_raw["연면적"] = pd.to_numeric(df[area_col], errors="coerce")
    df_raw["U"] = U
    df_raw["W"] = W
    df_raw["V"] = V

    # 숫자형 검증 (전체 NaN = 오류)
    for col in ["U", "W", "V", "연면적"]:
        if df_raw[col].notna().sum() == 0:
            st.error(f"❌ '{col}' 값이 모두 NaN 입니다. 원본 파일을 확인하세요.")
            return None

    return df_raw


# ============================================================
# (6) 다중 연도 파일 로딩
# ============================================================

def _extract_year_from_filename(filename: str):
    """파일명에서 20xx 년도 추출"""
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def load_all_years(upload_folder: str):
    """
    업로드 폴더 안의 연도별 파일들을 읽어:
        ({year: df_raw}, [에러메시지]) 반환.

    요구사항:
        - 실패한 연도는 완전히 제외
        - 오류는 errors 리스트에 모두 보관
        - app.py에서 get_year_to_raw()가 이 결과를 그대로 받아서 사용
    """

    year_to_raw = {}
    errors = []

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

        df_raw = load_energy_raw_for_analysis(path)
        if df_raw is None:
            errors.append(f"{year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    # 연도 정렬
    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda x: x[0]))

    if not year_to_raw:
        errors.append("유효한 연도 파일을 하나도 불러오지 못했습니다.")

    return year_to_raw, errors
