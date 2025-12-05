import os
import re
import pandas as pd
import streamlit as st


# ============================================================
# 공통 유틸: 컬럼 탐색
# ============================================================
def _find_column_by_keywords(df, keywords, required=True, col_label=""):
    """
    df.columns 중에서 keywords 리스트의 모든 키워드를 포함하는 첫 컬럼을 반환.
    못 찾으면 None (required=False) 또는 에러 메시지 출력 후 None.
    """
    cols = df.columns.tolist()
    target = None

    for c in cols:
        label = str(c)
        if all(k in label for k in keywords):
            target = c
            break

    if target is None and required:
        # 여기서는 st.stop 대신 None 반환만 하고,
        # 상위 로직에서 시트/연도별로 건너뛰도록 함
        st.warning(f"⚠ '{col_label}' 컬럼을 찾지 못했습니다. (키워드: {keywords}, cols={list(df.columns)})")
        return None

    return target


def _pick_org_column(df, used_cols):
    """
    기관명 컬럼 후보 찾기:
    1) 컬럼명에 기관/소속/사업장/시설/구분 등이 들어간 것 우선
    2) 그래도 없으면 숫자가 아닌(object) 컬럼 중, used_cols에 포함되지 않는 첫 컬럼
    """
    name_keywords = ["기관", "소속", "사업장", "시설", "구분", "사무소", "부서"]

    # 1차: 컬럼명 기반
    for c in df.columns:
        label = str(c)
        if any(k in label for k in name_keywords):
            if c not in used_cols:
                return c

    # 2차: dtype 기반 (object형 첫 컬럼)
    for c in df.columns:
        if c in used_cols:
            continue
        if df[c].dtype == "object":
            return c

    return None


def _pick_facility_column(df, used_cols):
    """
    시설구분 컬럼(의료/복지/기타)을 찾는다.
    없으면 None 반환.
    """
    facility_keywords = ["시설구분", "시설 구분", "시설유형", "시설종류", "시설", "분류", "유형"]

    for c in df.columns:
        label = str(c)
        if any(k in label for k in facility_keywords):
            if c not in used_cols:
                return c

    return None


# ============================================================
# 1) 단일 엑셀파일을 df_raw로 정제
# ============================================================
def load_energy_raw_for_analysis(path: str):
    """
    업로드된 연도별 '에너지 사용량관리.xlsx' 파일을 불러와
    df_raw 형태로 정제하여 리턴한다.
    - 워크북의 모든 시트를 검사하며,
    - 에너지 사용량 / 연면적 컬럼을 찾을 수 있는 시트를 선택한다.
    """

    try:
        # 모든 시트를 dict 형태로 로드
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        st.error(f"❌ 파일을 읽는 데 실패했습니다: {path} ({e})")
        return None

    selected_df = None
    selected_info = None

    # --------------------------------------------------------
    # 각 시트를 돌면서 "에너지사용량 / 연면적"이 있는 시트를 찾는다
    # --------------------------------------------------------
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        df.columns = df.columns.map(lambda x: str(x).strip())

        # 필수 컬럼 후보 탐색
        u_col = _find_column_by_keywords(
            df, ["에너지", "사용"], required=False, col_label="에너지 사용량(U)"
        )
        area_col = _find_column_by_keywords(
            df, ["면적"], required=False, col_label="연면적"
        )

        # 둘 다 있어야 유효한 시트로 인정
        if u_col is None or area_col is None:
            continue

        # 여기까지 왔으면 이 시트를 후보로 사용
        selected_df = df
        selected_info = (sheet_name, u_col, area_col)
        break

    if selected_df is None:
        st.error(
            f"❌ '{os.path.basename(path)}'에서 에너지 사용량(U)/연면적 컬럼을 모두 찾을 수 있는 시트를 발견하지 못했습니다."
        )
        return None

    sheet_name, U_col, area_col = selected_info
    df = selected_df

    # --------------------------------------------------------
    # 기관명 / 시설구분 / 기타 컬럼 매핑
    # --------------------------------------------------------
    used_cols = {U_col, area_col}

    # 기관명
    org_col = _pick_org_column(df, used_cols)
    if org_col is None:
        st.error(
            f"❌ '{os.path.basename(path)}' ({sheet_name})에서 기관명에 해당하는 컬럼을 찾지 못했습니다.\n"
            f"현재 컬럼 목록: {list(df.columns)}"
        )
        return None
    used_cols.add(org_col)

    # 면적당 에너지 사용비율(V) (있으면 사용)
    V_col = _find_column_by_keywords(
        df, ["면적", "에너지"], required=False, col_label="면적대비 에너지사용비율(V)"
    )
    if
