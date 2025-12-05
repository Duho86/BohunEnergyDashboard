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
    못 찾으면 None (required=False) 또는 경고 메시지 출력 후 None.
    """
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
    기관명 컬럼 후보 찾기:

    1) 컬럼명에 기관/소속/사업장/시설/구분 등이 들어간 것 우선
    2) 그래도 없으면 숫자가 아닌(object) 컬럼 중, used_cols에 포함되지 않는 첫 컬럼
    3) 그래도 없으면 used_cols에 포함되지 않는 '아무 컬럼이나' 첫 컬럼을 사용 (최종 fallback)
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

    # 3차: 최종 fallback - used_cols를 제외한 어떤 컬럼이든 하나 사용
    for c in df.columns:
        if c not in used_cols:
            st.warning(
                f"⚠ 기관명 컬럼을 추론할 수 없어 '{c}' 컬럼을 기관명으로 임시 사용합니다. "
                f"실제 엑셀 헤더를 확인해 주세요."
            )
            return c

    # 정말 모든 컬럼이 used_cols 안에만 있는 극단적 상황
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
    if V_col:
        used_cols.add(V_col)

    # 평균 에너지 사용량(W) (있으면 사용)
    W_col = _find_column_by_keywords(
        df, ["평균", "에너지"], required=False, col_label="평균 에너지 사용량(W)"
    )
    if W_col:
        used_cols.add(W_col)

    # 시설구분 (의료/복지/기타)
    facility_col = _pick_facility_column(df, used_cols)
    if facility_col:
        used_cols.add(facility_col)

    # --------------------------------------------------------
    # 필요한 컬럼만 추출 및 리네이밍
    # --------------------------------------------------------
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

    # 시설구분이 없으면 일단 기타시설로 채움 (경고 메시지)
    if "시설구분" not in df_raw.columns:
        st.warning(
            f"⚠ '{os.path.basename(path)}' ({sheet_name})에서 시설구분 컬럼을 찾지 못했습니다. "
            "모든 행을 '기타시설'로 처리합니다."
        )
        df_raw["시설구분"] = "기타시설"

    # --------------------------------------------------------
    # 숫자형 변환
    # --------------------------------------------------------
    numeric_cols = ["에너지사용량", "연면적", "면적대비사용비율", "평균에너지사용량"]
    for c in numeric_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

            if df_raw[c].notna().sum() == 0:
                st.error(
                    f"❌ '{os.path.basename(path)}' ({sheet_name})의 '{c}' 컬럼을 숫자로 변환할 수 없습니다. "
                    "데이터 형식을 확인해주세요."
                )
                return None

    return df_raw


# ============================================================
# 2) 다중 연도 파일 관리
# ============================================================
def extract_year_from_filename(filename: str):
    """
    파일명에서 2021, 2022 같은 4자리 숫자를 추출하여 연도로 사용.
    """
    m = re.search(r"(20\d{2})", filename)
    if m:
        return int(m.group(1))
    return None


def load_all_years(upload_folder: str):
    """
    저장된 파일들을 모두 불러와
    {연도: df_raw} 형태로 dict 반환.
    연도별로 로딩 실패 시, 해당 연도는 건너뛰고 경고 메시지만 출력.
    """
    year_to_raw = {}

    if not os.path.exists(upload_folder):
        return year_to_raw

    for filename in os.listdir(upload_folder):
        if not filename.endswith(".xlsx"):
            continue

        year = extract_year_from_filename(filename)
        if year is None:
            st.warning(f"⚠ 파일명에서 연도를 추출하지 못해 건너뜀: {filename}")
            continue

        path = os.path.join(upload_folder, filename)
        df_raw = load_energy_raw_for_analysis(path)

        if df_raw is None:
            st.error(f"❌ {year}년 파일 로딩 실패: {filename}")
            continue

        year_to_raw[year] = df_raw

    # 연도 순 정렬
    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda x: x[0]))

    if len(year_to_raw) == 0:
        st.warning("⚠ 업로드된 연도별 에너지 사용량 파일에서 유효한 데이터를 찾지 못했습니다.")
    return year_to_raw
