import os
import re
import pandas as pd
import streamlit as st


# ============================================================
# 1. 단일 연도 파일 → df_raw 생성
# ============================================================

def load_energy_raw_for_analysis(path: str):
    """업로드된 연도별 '에너지 사용량관리.xlsx' 파일을 df_raw 형태로 정제.

    ⚠ 이 함수는 2024년 예시 파일의 구조를 *사양서*로 간주한다.

    - 기관명           : '소속기관명'
    - 시설구분(사업군)  : '사업군'  (본사/의료/복지/기타 등)
    - 연면적           : '연면적/설비용량'
    - 월별 사용량      : '1월' ~ '12월'
    - U(연간 에너지 사용량 합계) : '연단위'  (없으면 1~12월 합계로 대체)
    - V(면적당 온실가스 배출량) : '면적당 온실가스\\n배출량'
    - W(평균 에너지 사용량)     : 1~12월 평균

    반환 df_raw 컬럼:
        ['기관명', '시설구분', '연면적', 'U', 'V', 'W']
    """

    try:
        # 0행은 "진행상태/사업군/소속기관명/…/1월/2월/…" 라벨이 들어 있으므로 header=1
        df = pd.read_excel(path, sheet_name=0, header=1)
    except Exception as e:
        st.error(f"❌ 파일 로딩 실패: {os.path.basename(path)} ({e})")
        return None

    df.columns = df.columns.map(str)

    # -----------------------------
    # 필수 컬럼 존재 여부 체크
    # -----------------------------
    required_cols = {
        "기관명": "소속기관명",
        "사업군": "사업군",
        "연면적": "연면적/설비용량",
        "V": "면적당 온실가스\n배출량",
    }

    for label, col in required_cols.items():
        if col not in df.columns:
            st.error(f"❌ {label}({col}) 컬럼을 찾지 못했습니다. 파일 헤더를 확인하세요.")
            return None

    # 월별 에너지 사용량 컬럼 (1월 ~ 12월)
    month_cols = [f"{m}월" for m in range(1, 13)]
    missing = [c for c in month_cols if c not in df.columns]
    if missing:
        st.error(
            "❌ 월별 에너지 사용량 컬럼(1월~12월) 중 일부가 없습니다. "
            "누락 컬럼: " + ", ".join(missing)
        )
        return None

    # -----------------------------
    # 숫자형 변환
    # -----------------------------
    numeric_cols = month_cols + [required_cols["연면적"], "연단위", required_cols["V"]]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 필수 숫자 컬럼 검증
    for col in [required_cols["연면적"], "연단위", required_cols["V"]]:
        if col in df.columns and df[col].notna().sum() == 0:
            st.error(f"❌ '{col}' 값이 모두 NaN 입니다. 원본 데이터를 확인하세요.")
            return None

    # -----------------------------
    # df_raw 구성
    # -----------------------------
    df_raw = pd.DataFrame()
    df_raw["기관명"] = df[required_cols["기관명"]]

    # 사업군 → 시설구분(의료시설/복지시설/기타시설) 매핑
    def _map_facility_group(x: str) -> str:
        x = str(x)
        if "의료" in x:
            return "의료시설"
        if "복지" in x:
            return "복지시설"
        # 본사/기타/기타사업 등은 모두 기타시설 처리
        return "기타시설"

    df_raw["시설구분"] = df[required_cols["사업군"]].map(_map_facility_group)

    df_raw["연면적"] = df[required_cols["연면적"]]

    # U: 연단위 합계를 우선 사용, 없거나 전부 NaN이면 1~12월 합계
    if "연단위" in df.columns and df["연단위"].notna().sum() > 0:
        df_raw["U"] = df["연단위"]
    else:
        df_raw["U"] = df[month_cols].sum(axis=1)

    # W: 1~12월 평균
    df_raw["W"] = df[month_cols].mean(axis=1)

    # V: 면적당 온실가스 배출량
    df_raw["V"] = df[required_cols["V"]]

    # 최종 숫자형 검증 (None/NaN만 있는 경우는 오류 처리)
    for col in ["U", "W", "V", "연면적"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        if df_raw[col].notna().sum() == 0:
            st.error(f"❌ '{col}' 값이 모두 NaN 입니다. 원본 데이터를 확인하세요.")
            return None

    return df_raw


# ============================================================
# 2. 다중 연도 파일 로딩
# ============================================================

def _extract_year_from_filename(filename: str):
    """파일명에서 4자리 연도(20xx)를 추출."""
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def load_all_years(upload_folder: str):
    """업로드 폴더에 있는 모든 연도 파일을 읽어 {연도: df_raw}, [오류메시지] 반환.

    - 파일명에서 연도 추출 실패 → errors에 기록 후 건너뜀
    - 개별 파일 로딩 실패(df_raw=None) → errors에 기록 후 건너뜀
    - 유효한 연도가 하나도 없으면 errors에 경고 추가
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
            errors.append(f"연도를 파일명에서 추출하지 못했습니다: {filename}")
            continue

        path = os.path.join(upload_folder, filename)

        df_raw = load_energy_raw_for_analysis(path)
        if df_raw is None:
            errors.append(f"{year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    # 연도 순으로 정렬
    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda x: x[0]))

    if not year_to_raw:
        errors.append("업로드된 연도별 에너지 사용량 파일에서 유효한 데이터를 찾지 못했습니다.")

    return year_to_raw, errors
