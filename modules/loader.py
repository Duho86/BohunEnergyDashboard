import os
import re
import pandas as pd
import streamlit as st


# ============================================================
# 공통 유틸: 컬럼 탐색
# ============================================================
def _find_column_by_keywords(df, keywords, required=True, col_label=""):
    cols = df.columns.tolist()
    target = None

    for c in cols:
        label = str(c)
        if all(k in label for k in keywords):
            target = c
            break

    if target is None and required:
        st.warning(
            f"⚠ '{col_label}' 컬럼을 찾지 못했습니다. "
            f"(키워드: {keywords}, cols={list(df.columns)})"
        )
        return None

    return target


def _pick_org_column(df, used_cols):
    """
    기관명 컬럼을 아래 우선순위로 선택한다:
    1) 컬럼명이 기관/소속/시설/구분/사업장/부서 등 포함
    2) dtype == object 인 첫 번째 컬럼
    3) used_cols 를 제외한 아무 컬럼이나 강제로 기관명으로 사용 (fallback)
    """
    name_keywords = ["기관", "소속", "사업장", "시설", "구분", "사무소", "부서"]

    # 1차: 키워드 기반 탐색
    for c in df.columns:
        if any(k in str(c) for k in name_keywords):
            if c not in used_cols:
                return c

    # 2차: object dtype 첫 컬럼
    for c in df.columns:
        if c in used_cols:
            continue
        if df[c].dtype == "object":
            return c

    # 3차: 최종 fallback → 어떤 경우에도 기관명 하나 선택
    for c in df.columns:
        if c not in used_cols:
            st.warning(
                f"⚠ 기관명 컬럼 자동 탐색 실패 → '{c}' 컬럼을 기관명으로 임시 사용합니다."
            )
            return c

    return None


def _pick_facility_column(df, used_cols):
    facility_keywords = ["시설구분", "시설 구분", "시설유형", "시설종류", "시설", "분류", "유형"]

    for c in df.columns:
        if any(k in str(c) for k in facility_keywords):
            if c not in used_cols:
                return c

    return None


# ============================================================
# 1) 단일 파일 로더
# ============================================================
def load_energy_raw_for_analysis(path: str):
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        st.error(f"❌ 파일 읽기 실패: {path} ({e})")
        return None

    selected_df = None
    selected_info = None

    # 유효한 시트를 찾는 과정
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        df.columns = df.columns.map(lambda x: str(x).strip())

        u_col = _find_column_by_keywords(df, ["에너지", "사용"], required=False)
        area_col = _find_column_by_keywords(df, ["면적"], required=False)

        if u_col and area_col:
            selected_df = df
            selected_info = (sheet_name, u_col, area_col)
            break

    if selected_df is None:
        st.error(
            f"❌ '{os.path.basename(path)}'에서 에너지사용량(U)/연면적 컬럼을 둘 다 갖는 시트를 찾지 못했습니다."
        )
        return None

    sheet_name, U_col, area_col = selected_info
    df = selected_df
    used_cols = {U_col, area_col}

    # 기관명 컬럼 선택 (fallback 포함)
    org_col = _pick_org_column(df, used_cols)
    if org_col is None:
        st.error(f"❌ '{os.path.basename(path)}' ({sheet_name})에서 기관명 컬럼을 찾지 못했습니다.")
        return None
    used_cols.add(org_col)

    # 선택적 컬럼
    V_col = _find_column_by_keywords(df, ["면적", "에너지"], required=False)
    if V_col:
        used_cols.add(V_col)

    W_col = _find_column_by_keywords(df, ["평균", "에너지"], required=False)
    if W_col:
        used_cols.add(W_col)

    facility_col = _pick_facility_column(df, used_cols)
    if facility_col:
        used_cols.add(facility_col)

    # 추출할 컬럼 목록
    use_cols = [org_col, U_col, area_col]
    if V_col:
        use_cols.append(V_col)
    if W_col:
        use_cols.append(W_col)
    if facility_col:
        use_cols.append(facility_col)

    df_raw = df[use_cols].copy()

    rename_map = {
        org_col: "기관명",
        U_col: "에너지사용량",
        area_col: "연면적",
    }
    if V_col:
        rename_map[V_col] = "면적대비사용비율"
    if W_col:
        rename_map[W_col] = "평균에너지사용량"
    if facility_col:
        rename_map[facility_col] = "시설구분"

    df_raw = df_raw.rename(columns=rename_map)

    # 시설구분 없으면 기본값
    if "시설구분" not in df_raw.columns:
        st.warning(
            f"⚠ '{os.path.basename(path)}' ({sheet_name})에서 시설구분 컬럼을 찾지 못했습니다. "
            f"'기타시설'로 처리합니다."
        )
        df_raw["시설구분"] = "기타시설"

    # 숫자형 변환
    for col in ["에너지사용량", "연면적", "면적대비사용비율", "평균에너지사용량"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    return df_raw


# ============================================================
# 2) 연도별 로더
# ============================================================
def extract_year_from_filename(filename: str):
    m = re.search(r"(20\d{2})", filename)
    if m:
        return int(m.group(1))
    return None


def load_all_years(upload_folder: str):
    year_to_raw = {}

    if not os.path.exists(upload_folder):
        return year_to_raw

    for filename in os.listdir(upload_folder):
        if not filename.endswith(".xlsx"):
            continue

        year = extract_year_from_filename(filename)
        if year is None:
            st.warning(f"⚠ 연도를 찾지 못해 건너뜀: {filename}")
            continue

        path = os.path.join(upload_folder, filename)
        df_raw = load_energy_raw_for_analysis(path)

        if df_raw is None:
            st.error(f"❌ {year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    return dict(sorted(year_to_raw.items(), key=lambda x: x[0]))
