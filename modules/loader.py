import os
import re
import pandas as pd
import streamlit as st

# -----------------------------
# 1) 컬럼 자동 탐색 함수
# -----------------------------
def find_column(df, keywords, required=True, col_label=""):
    """
    df에서 특정 키워드들을 포함한 컬럼을 자동으로 탐색한다.
    keywords: ["에너지", "사용량"] 같이 리스트 형태
    required = True이면 컬럼이 없을 때 오류 출력
    """
    cols = df.columns.tolist()
    target = None

    for c in cols:
        label = str(c)
        if all(k in label for k in keywords):
            target = c
            break

    if required and target is None:
        st.error(f"❌ '{col_label}' 컬럼을 찾지 못했습니다. (키워드: {keywords})")
        st.stop()

    return target


# -----------------------------
# 2) 단일 엑셀파일을 df_raw로 정제
# -----------------------------
def load_energy_raw_for_analysis(path: str):
    """
    업로드된 연도별 '에너지 사용량관리.xlsx' 파일을 불러와
    df_raw 형태로 정제하여 리턴한다.
    이 df_raw는 analyzer.py의 모든 분석 로직의 기반이 된다.
    """

    try:
        df = pd.read_excel(path)
    except:
        st.error(f"❌ 파일을 읽는 데 실패했습니다: {path}")
        return None

    # 기본 전처리
    df.columns = df.columns.map(lambda x: str(x).strip())

    # 기관명 컬럼(C열이라는 전제) 자동 탐색
    # 실제 엑셀 시트 구조를 보면 C열이 '구분' 또는 '기관명' 역할을 수행
    possible_org_cols = [c for c in df.columns if "구분" in c or "기관" in c or "소속" in c]

    if len(possible_org_cols) == 0:
        st.error("❌ 기관명(구분) 컬럼을 찾지 못했습니다.")
        st.stop()

    org_col = possible_org_cols[0]

    # -------------------------
    # 핵심 4개 컬럼 자동 매핑
    # -------------------------
    # ① 에너지 사용량(U)
    U_col = find_column(df,
                        keywords=["에너지", "사용"],
                        required=True,
                        col_label="에너지 사용량(U)")

    # ② 연면적
    area_col = find_column(df,
                           keywords=["면적"],
                           required=True,
                           col_label="연면적")

    # ③ 면적당 에너지(또는 온실가스) 사용비율(V)
    # 시트2·3의 표 기준으로 "면적대비", "비율" 등의 표현이 있음
    V_col = find_column(df,
                        keywords=["면적", "비"],
                        required=False,
                        col_label="면적대비 에너지사용비율(V)")

    # ④ 평균 에너지 사용량(W)
    W_col = find_column(df,
                        keywords=["평균", "에너지"],
                        required=False,
                        col_label="평균 에너지 사용량(W)")

    # -------------------------
    # 필요한 컬럼만 추출
    # -------------------------
    use_cols = [org_col, U_col, area_col]
    if V_col:
        use_cols.append(V_col)
    if W_col:
        use_cols.append(W_col)

    df_raw = df[use_cols].copy()

    # 컬럼명 정규화
    df_raw = df_raw.rename(columns={
        org_col: "기관명",
        U_col: "에너지사용량",
        area_col: "연면적",
        V_col: "면적대비사용비율" if V_col else None,
        W_col: "평균에너지사용량" if W_col else None
    })

    # 숫자형 변환
    numeric_cols = ["에너지사용량", "연면적", "면적대비사용비율", "평균에너지사용량"]
    for c in numeric_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

            # 변환 실패 시(전체 NaN) 오류 처리
            if df_raw[c].notna().sum() == 0:
                st.error(f"❌ '{c}' 컬럼을 숫자로 변환할 수 없습니다. 데이터 형식을 확인해주세요.")
                st.stop()

    return df_raw


# -----------------------------
# 3) 다중 연도 파일 관리
# -----------------------------
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
    """
    year_to_raw = {}

    for filename in os.listdir(upload_folder):
        if not filename.endswith(".xlsx"):
            continue

        year = extract_year_from_filename(filename)
        if year is None:
            continue

        path = os.path.join(upload_folder, filename)
        df_raw = load_energy_raw_for_analysis(path)

        if df_raw is not None:
            year_to_raw[year] = df_raw

    # 연도 순 정렬
    year_to_raw = dict(sorted(year_to_raw.items(), key=lambda x: x[0]))

    if len(year_to_raw) == 0:
        st.warning("⚠ 업로드된 연도별 에너지 사용량 파일이 없습니다.")
    return year_to_raw

