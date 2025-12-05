import os
import re
import pandas as pd
import streamlit as st

# -----------------------------
# Helper functions for column detection
# -----------------------------

def _find_col(df, required_substrings, col_label, required=True):
    """
    지정한 문자열들을 모두 포함하는 헤더를 가진 컬럼을 찾는다.
    - required_substrings: ["에너지", "사용량"] 이런 식의 리스트
    - 못 찾으면, required=True일 때 st.error 메시지 출력 후 None 반환
    """
    for c in df.columns:
        label = str(c)
        if all(sub in label for sub in required_substrings):
            return c
    if required:
        st.error(
            f"❌ {col_label} 컬럼을 찾지 못했습니다. "
            f"(필요 포함 문자열: {required_substrings}, 실제 컬럼: {list(df.columns)})"
        )
    return None


def _find_org_col(df):
    """
    기관명(구분) 컬럼 탐색:
    - 소속기관명, 소속기관, 기관명, 구분 등 이름에 기반해서만 찾는다.
    - 못 찾으면 에러 출력 후 None.
    """
    candidates = [
        c
        for c in df.columns
        if any(kw in str(c) for kw in ["소속기관명", "소속기관", "기관명", "구분"])
    ]
    if candidates:
        return candidates[0]

    st.error("❌ 기관명(구분) 컬럼을 찾지 못했습니다. (예: '소속기관명', '소속기관', '기관명')")
    return None


def _find_facility_col(df):
    """
    시설구분 컬럼 탐색:
    - 시설구분, 시설 구분, 시설유형, 용도(구분) 등 이름 기반.
    - 시트2/3에 필수이므로 못 찾으면 에러.
    """
    candidates = [
        c
        for c in df.columns
        if any(kw in str(c) for kw in ["시설구분", "시설 구분", "시설유형", "용도", "용도구분"])
    ]
    if candidates:
        return candidates[0]

    st.error("❌ 시설구분 컬럼을 찾지 못했습니다. (예: '시설구분', '시설유형')")
    return None


# -----------------------------
# Core loader
# -----------------------------

def load_energy_raw_for_analysis(path: str) -> pd.DataFrame | None:
    """
    단일 에너지 사용량 엑셀 파일을 읽어서 df_raw를 반환.

    - df_raw는 원본 컬럼을 모두 유지하되, 다음 표준 컬럼을 추가한다.
      * 기관명
      * 시설구분
      * U (에너지 사용량)
      * V (면적당 온실가스 배출량)
      * W (평균 에너지 사용량)
      * 연면적

    - 컬럼은 열 인덱스가 아니라 '헤더 문자열' 기반으로 탐색한다.
    - 필수 컬럼을 찾지 못하면 st.error를 띄우고 None을 반환한다.
    """
    try:
        # 기본 가정: 첫 번째 시트에 분석용 데이터가 있음
        df = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        st.error(f"❌ 엑셀 파일을 읽는 데 실패했습니다: {path} ({e})")
        return None

    if df.empty:
        st.error(f"❌ 엑셀 파일에 데이터가 없습니다: {path}")
        return None

    # 공백 제거
    df.columns = df.columns.map(lambda x: str(x).strip())

    # 기관명 / 시설구분
    org_col = _find_org_col(df)
    facility_col = _find_facility_col(df)

    if org_col is None or facility_col is None:
        return None

    # U / V / W / 연면적 컬럼 이름 탐색 (문자열 기준)
    U_src = _find_col(
        df,
        required_substrings=["에너지", "사용량"],
        col_label="에너지 사용량(U)",
        required=True,
    )
    V_src = _find_col(
        df,
        required_substrings=["면적당", "온실가스"],
        col_label="면적당 온실가스 배출량(V)",
        required=True,
    )
    W_src = _find_col(
        df,
        required_substrings=["평균", "에너지", "사용량"],
        col_label="평균 에너지 사용량(W)",
        required=True,
    )
    area_src = _find_col(
        df,
        required_substrings=["연면적"],
        col_label="연면적",
        required=True,
    )

    # 필수 컬럼 중 하나라도 없으면 이 파일은 분석 대상에서 제외
    if any(c is None for c in [U_src, V_src, W_src, area_src]):
        return None

    # 표준 컬럼 생성 (원본 컬럼은 그대로 두고 추가)
    df["기관명"] = df[org_col]
    df["시설구분"] = df[facility_col]
    df["U"] = df[U_src]
    df["V"] = df[V_src]
    df["W"] = df[W_src]
    df["연면적"] = df[area_src]

    # 숫자형 변환 (필수)
    for col in ["U", "V", "W", "연면적"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() == 0:
            st.error(
                f"❌ '{col}' 컬럼의 값을 숫자로 변환하지 못했습니다. "
                f"원본 데이터를 확인해 주세요. (파일: {os.path.basename(path)})"
            )
            return None

    return df


# -----------------------------
# 다중 연도 로더
# -----------------------------

def _extract_year_from_filename(filename: str) -> int | None:
    """
    파일명에서 4자리 연도(20xx)를 추출.
    예: '에너지사용량_2024.xlsx' → 2024
    """
    m = re.search(r"(20\d{2})", filename)
    if not m:
        return None
    return int(m.group(1))


def load_all_years(upload_folder: str):
    """
    업로드 폴더 내의 .xlsx 파일들을 모두 읽어
    ({연도: df_raw}, [오류 메시지 리스트]) 를 반환한다.

    - 각 연도 파일은 load_energy_raw_for_analysis()로 처리한다.
    - 특정 연도에서 로딩에 실패해도 다른 연도는 계속 진행한다.
    """
    year_to_raw: dict[int, pd.DataFrame] = {}
    errors: list[str] = []

    if not os.path.exists(upload_folder):
        return year_to_raw, errors

    for filename in os.listdir(upload_folder):
        if not filename.lower().endswith(".xlsx"):
            continue

        year = _extract_year_from_filename(filename)
        if year is None:
            errors.append(f"파일명에서 연도를 찾지 못해 건너뜀: {filename}")
            continue

        path = os.path.join(upload_folder, filename)
        df_raw = load_energy_raw_for_analysis(path)
        if df_raw is None:
            errors.append(f"{year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    # 연도 오름차순 정렬
    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda kv: kv[0]))
    return year_to_raw, errors
