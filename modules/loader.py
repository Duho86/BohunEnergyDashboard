import os
import re
import pandas as pd
import streamlit as st

# ============================================================
# 공통 유틸: 컬럼 탐색 (헤더 이름 기반 · 인덱스 사용 금지)
# ============================================================

def _find_col(df, must_contain: list[str], label: str, required: bool = True):
    """
    df.columns 중에서 must_contain 에 포함된 문자열을 모두 포함하는
    첫 번째 컬럼을 반환한다.
    - 한 글자라도 빠지면 안 됨 (AND 조건)
    - required=True 인데 못 찾으면 st.error() 띄우고 None 반환
    """
    for c in df.columns:
        header = str(c)
        if all(sub in header for sub in must_contain):
            return c

    if required:
        st.error(
            f"❌ '{label}' 컬럼을 찾지 못했습니다.\n"
            f"  · 필요 포함 문자열: {must_contain}\n"
            f"  · 실제 컬럼 목록: {list(df.columns)}"
        )
    return None


def _find_org_col(df):
    """
    기관명(구분) 컬럼: 예시 엑셀 기준으로
      - '소속기관명', '소속기관', '기관명', '구분' 등 헤더 문자열을 사용
    """
    for c in df.columns:
        header = str(c)
        if any(k in header for k in ["소속기관명", "소속기관", "기관명", "구분"]):
            return c

    st.error(
        "❌ 기관명(구분) 컬럼을 찾지 못했습니다.\n"
        "  · 예: '소속기관명', '소속기관', '기관명', '구분' 등의 헤더가 필요합니다."
    )
    return None


def _find_facility_col(df):
    """
    시설구분 컬럼: 의료/복지/기타 구분용
      - '시설구분', '시설 구분', '시설유형', '용도(구분)' 등 헤더 문자열을 사용
    """
    for c in df.columns:
        header = str(c)
        if any(k in header for k in ["시설구분", "시설 구분", "시설유형", "용도", "용도(구분)"]):
            return c

    st.error(
        "❌ 시설구분 컬럼을 찾지 못했습니다.\n"
        "  · 예: '시설구분', '시설유형', '용도(구분)' 등의 헤더가 필요합니다."
    )
    return None


# ============================================================
# 단일 파일 → df_raw
# ============================================================

def load_energy_raw_for_analysis(path: str) -> pd.DataFrame | None:
    """
    하나의 《에너지 사용량관리.xlsx》 파일을 읽어 df_raw 생성.

    ✔ 요구사항 반영:
      - df_std 사용 금지, df_raw 만 사용.
      - U/V/W/연면적은 열 인덱스가 아니라 '헤더 문자열'로만 식별.
      - 찾지 못한 경우 임의 계산 금지, 명시적 에러 후 해당 연도 분석 제외.
      - 숫자형 변환 후 전체 NaN 이면 에러 처리.

    df_raw에 반드시 포함되는 컬럼:
      - 기관명
      - 시설구분
      - U  (에너지 사용량)
      - V  (면적당 온실가스 배출량)
      - W  (평균 에너지 사용량)
      - 연면적
    """
    try:
        # 예시 엑셀 기준으로 분석용 데이터는 첫 번째 시트에 있다고 가정
        df = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        st.error(f"❌ 엑셀 파일을 읽는 데 실패했습니다: {os.path.basename(path)} ({e})")
        return None

    if df is None or df.empty:
        st.error(f"❌ 엑셀 파일에 데이터가 없습니다: {os.path.basename(path)}")
        return None

    # 헤더 공백 제거
    df.columns = df.columns.map(lambda x: str(x).strip())

    # 1) 기관명 / 시설구분
    org_col = _find_org_col(df)
    facility_col = _find_facility_col(df)
    if org_col is None or facility_col is None:
        # 에러 메시지는 각각 함수에서 이미 출력
        return None

    # 2) U/V/W/연면적 컬럼 이름 식별 (문자열 기반)
    #   예시:
    #   - "에너지 사용량(합계)"     → U
    #   - "면적당 온실가스 배출량" → V
    #   - "평균 에너지 사용량(W)"  → W
    #   - "연면적"                 → 연면적
    U_src = _find_col(
        df,
        must_contain=["에너지", "사용량"],
        label="에너지 사용량(U)",
        required=True,
    )
    V_src = _find_col(
        df,
        must_contain=["면적당", "온실가스"],
        label="면적당 온실가스 배출량(V)",
        required=True,
    )
    W_src = _find_col(
        df,
        must_contain=["평균", "에너지", "사용량"],
        label="평균 에너지 사용량(W)",
        required=True,
    )
    area_src = _find_col(
        df,
        must_contain=["연면적"],
        label="연면적",
        required=True,
    )

    # 필수 컬럼 중 하나라도 없으면 이 파일은 분석에서 제외
    if any(c is None for c in [U_src, V_src, W_src, area_src]):
        return None

    # 3) 표준 컬럼 생성 (원본 컬럼은 그대로 두고, 분석에서 쓸 표준 컬럼만 추가)
    df_raw = df.copy()
    df_raw["기관명"] = df_raw[org_col]
    df_raw["시설구분"] = df_raw[facility_col]
    df_raw["U"] = df_raw[U_src]
    df_raw["V"] = df_raw[V_src]
    df_raw["W"] = df_raw[W_src]
    df_raw["연면적"] = df_raw[area_src]

    # 4) 숫자형 변환 + 전체 NaN 체크
    for col in ["U", "V", "W", "연면적"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        if df_raw[col].notna().sum() == 0:
            st.error(
                f"❌ '{os.path.basename(path)}' 파일에서 '{col}' 컬럼을 숫자로 변환한 결과가 모두 NaN 입니다.\n"
                "  · 원본 데이터 형식을 다시 확인해 주세요."
            )
            return None

    return df_raw


# ============================================================
# 다중 연도 로더
# ============================================================

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
    업로드 폴더 내 .xlsx 파일들을 모두 읽어
    (year_to_raw, errors) 형태로 반환한다.

    - year_to_raw: {연도(int): df_raw(DataFrame)}
    - errors:      [로딩 중 발생한 경고/오류 메시지 문자열]

    ✔ 요구사항 반영:
      - 특정 연도 파일에 문제가 있어도 전체 앱이 죽지 않도록,
        해당 연도만 제외하고 오류 메시지는 화면 상단 Expander로 전달.
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
            # 세부 에러 메시지는 load_energy_raw_for_analysis 에서 이미 st.error로 출력됨
            errors.append(f"{year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    # 연도 정렬
    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda kv: kv[0]))

    return year_to_raw, errors
