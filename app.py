import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

# ============================================================
# 0. 기본 설정
# ============================================================

st.set_page_config(
    page_title="공단 에너지 사용량 분석",
    layout="wide",
)

st.title("공단 에너지 사용량 분석 대시보드")

# ============================================================
# 1. 데이터 로딩 관련 유틸
# ============================================================

@st.cache_data
def load_raw_from_excel(uploaded_file: bytes, year: int) -> pd.DataFrame:
    """
    백데이터 엑셀(또는 CSV/JSON)을 그대로 읽어와서
    연도별 df_raw 형태로 통일한다.

    반드시 포함되어야 하는 컬럼(이름은 엑셀과 맞추면 됨):
    - '사업군'          : 본사 / 의료 / 복지 / 기타 등
    - '소속기관명'      : 본사, 중앙보훈병원, 부산보훈병원 ...
    - '연면적/설비용량' : 숫자(연면적)
    - '연단위'          : 연간 에너지 사용량 (kWh 또는 환산값)
    - (선택) '연료', '단위'  : 상세용
    """
    # 엑셀/CSV 구분해서 처리 (여기서는 엑셀 기준)
    df = pd.read_excel(uploaded_file)

    # 컬럼 표준화 (필요하면 여기 이름만 바꾸면 됨)
    rename_map = {
        "연면적/설비용량": "연면적",
        "소속기구명": "소속기관명",  # 이미 소속기관명으로 되어 있으면 무시됨
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 숫자 컬럼 정리
    for col in ["연면적", "연단위"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 연도 컬럼 추가
    df["기준연도"] = year

    return df


def concat_years(dfs_by_year: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """여러 연도의 df_raw를 하나로 합친다."""
    valid = [df.assign(기준연도=year) for year, df in dfs_by_year.items() if df is not None]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def add_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 df_raw에 분석에 필요한 파생 컬럼들을 추가.

    - 시설구분_대분류 : 사업군 -> 의료시설/복지시설/기타시설 매핑
    - 기관 레벨 집계 등에 사용할 기본 컬럼 정리
    """
    df = df.copy()

    # 사업군 기준으로 시설 대분류 생성
    def map_business_to_type(x: str) -> str:
        if pd.isna(x):
            return "기타시설"
        x = str(x)
        if "의료" in x:
            return "의료시설"
        if "복지" in x:
            return "복지시설"
        # 본사/교육원/요양원 등 나머지
        return "기타시설"

    df["시설구분_대분류"] = df.get("시설구분_대분류")  # 있을 수도 있으니
    if "시설구분_대분류" not in df.columns or df["시설구분_대분류"].isna().all():
        df["시설구분_대분류"] = df["사업군"].apply(map_business_to_type)

    # 연면적/연단위 숫자화
    for col in ["연면적", "연단위"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 2. 분석 계산 로직 (엑셀 수식 1:1 복원)
# ============================================================

def compute_org_level_table(
    all_data: pd.DataFrame,
    base_year: int,
) -> pd.DataFrame:
    """
    2. 소속기구별 표를 만드는 함수.
    - all_data : 여러 연도가 합쳐진 df_raw (기준연도 컬럼 포함)
    - base_year: 기준연도 (예: 2024)

    엑셀 '에너지 사용량 분석' 시트의 2번 표와 동일한 결과를 만든다.
    """

    if all_data.empty:
        return pd.DataFrame()

    df = add_classification_columns(all_data)

    # 현재 연도 / 전년 / 과거 3개년 데이터 분리
    df_cur = df[df["기준연도"] == base_year].copy()
    df_prev = df[df["기준연도"] == base_year - 1].copy()
    df_3y = df[df["기준연도"].between(base_year - 3, base_year - 1)].copy()

    if df_cur.empty:
        return pd.DataFrame()

    # 기관별 집계 (연료별 row들을 기관별로 합침)
    def agg_per_org(d: pd.DataFrame) -> pd.DataFrame:
        g = (
            d.groupby("소속기관명", dropna=False)
            .apply(
                lambda g_: pd.Series(
                    {
                        "시설구분": g_["시설구분_대분류"].dropna().iloc[0]
                        if not g_["시설구분_대분류"].dropna().empty
                        else "기타시설",
                        "연면적": g_["연면적"].dropna().iloc[0]
                        if not g_["연면적"].dropna().empty
                        else np.nan,
                        "에너지사용량": g_["연단위"].sum(),
                    }
                )
            )
            .reset_index()
        )
        return g

    cur_org = agg_per_org(df_cur)
    total_energy_cur = cur_org["에너지사용량"].sum()

    # 전년/3개년용 기관별 에너지 사용량
    prev_org = agg_per_org(df_prev) if not df_prev.empty else pd.DataFrame(columns=cur_org.columns)
    three_year_org = (
        df_3y.groupby(["기준연도", "소속기관명"], dropna=False)["연단위"]
        .sum()
        .reset_index()
    )

    # 기관별 3개년 평균 계산
    avg_3y_map = {}
    for org in cur_org["소속기관명"]:
        hist = three_year_org[three_year_org["소속기관명"] == org]
        if hist.empty:
            avg_3y_map[org] = np.nan
        else:
            avg_3y_map[org] = hist["연단위"].mean()

    # 시설구분별 면적대비 에너지 사용비율 평균을 구하기 위해
    # 먼저 기관별 면적대비 비율을 계산
    cur_org["면적대비에너지사용비율"] = (
        cur_org["연면적"] / cur_org["에너지사용량"] * 100
    )

    facility_type_avg = (
        cur_org.groupby("시설구분")["면적대비에너지사용비율"].mean()
    )  # 의료/복지/기타별 평균 (%)

    # 각 기관별 지표 계산
    rows = []
    change_rates_3y_for_total_avg = []

    for _, r in cur_org.iterrows():
        org = r["소속기관명"]
        facility_type = r["시설구분"]
        area = r["연면적"]
        energy_cur = r["에너지사용량"]

        # 면적대비 에너지 사용비율 (%)
        area_ratio = area / energy_cur * 100 if energy_cur > 0 else np.nan

        # 에너지 사용 비중 (%)
        share = energy_cur / total_energy_cur * 100 if total_energy_cur > 0 else np.nan

        # 3개년 평균 대비 증감률
        avg_3y = avg_3y_map.get(org, np.nan)
        if pd.isna(avg_3y) or avg_3y == 0:
            change_vs_3y = np.nan
        else:
            change_vs_3y = (energy_cur - avg_3y) / avg_3y * 100

        # 시설별 평균 면적 대비 에너지 사용비율 (%)
        type_avg = facility_type_avg.get(facility_type, np.nan)
        if pd.isna(type_avg) or type_avg == 0:
            area_ratio_vs_type_avg = np.nan
        else:
            area_ratio_vs_type_avg = area_ratio / type_avg * 100

        rows.append(
            {
                "구 분": org,
                "시설구분": facility_type,
                "연면적": area,
                "에너지 사용량": energy_cur,
                "면적대비 에너지 사용비율(%)": area_ratio,
                "에너지 사용 비중(%)": share,
                "3개년 평균 에너지 사용량 대비 증감률(%)": change_vs_3y,
                "시설별 평균 면적 대비 에너지 사용비율(%)": area_ratio_vs_type_avg,
            }
        )

        if not pd.isna(change_vs_3y):
            change_rates_3y_for_total_avg.append(change_vs_3y)

    org_table = pd.DataFrame(rows)

    # 합계 행 (엑셀처럼 "행들의 평균"을 사용하는 컬럼은 평균으로)
    total_row = {
        "구 분": "합 계",
        "시설구분": "",
        "연면적": org_table["연면적"].sum(),
        "에너지 사용량": org_table["에너지 사용량"].sum(),
        # 면적대비 에너지 사용비율 -> 행별 값의 평균 (엑셀 1.03%)
        "면적대비 에너지 사용비율(%)": org_table["면적대비 에너지 사용비율(%)"].mean(),
        "에너지 사용 비중(%)": 100.0,
        # 3개년 평균 대비 증감률 -> 행별 값의 평균 (엑셀 18.59%)
        "3개년 평균 에너지 사용량 대비 증감률(%)": np.nanmean(change_rates_3y_for_total_avg)
        if change_rates_3y_for_total_avg
        else np.nan,
        # 시설별 평균 면적 대비 에너지 사용비율 -> 100%
        "시설별 평균 면적 대비 에너지 사용비율(%)": 100.0,
    }

    org_table = pd.concat(
        [org_table, pd.DataFrame([total_row])], ignore_index=True
    )

    return org_table


def compute_corp_summary(
    org_table: pd.DataFrame,
    all_data: pd.DataFrame,
    base_year: int,
) -> pd.DataFrame:
    """
    1. 공단 전체 기준 표 생성.
    - org_table: 바로 위 함수의 결과 전체(필터 적용 전 기준)
    - all_data : concat_years 한 전체 df_raw
    """

    if org_table.empty or all_data.empty:
        return pd.DataFrame()

    df = add_classification_columns(all_data)

    df_cur = df[df["기준연도"] == base_year].copy()
    df_prev = df[df["기준연도"] == base_year - 1].copy()
    df_3y = df[df["기준연도"].between(base_year - 3, base_year - 1)].copy()

    # 전체 에너지
    total_cur = df_cur["연단위"].sum()
    total_prev = df_prev["연단위"].sum() if not df_prev.empty else np.nan

    # 전년 대비 증감률
    if pd.isna(total_prev) or total_prev == 0:
        change_vs_prev = np.nan
    else:
        change_vs_prev = (total_cur - total_prev) / total_prev * 100

    # 3개년 평균 대비 증감률 (전체)
    if df_3y.empty:
        change_vs_3y = np.nan
    else:
        total_3y_avg = (
            df_3y.groupby("기준연도")["연단위"].sum().mean()
        )
        change_vs_3y = (
            (total_cur - total_3y_avg) / total_3y_avg * 100
            if total_3y_avg > 0
            else np.nan
        )

    # 시설구분별 면적대비 평균 에너지 사용비율
    # -> 2번 표에서 이미 계산한 "면적대비 에너지 사용비율(%)" 의
    #    시설구분별 평균값을 재사용
    org_only = org_table[org_table["구 분"] != "합 계"].copy()
    type_avg = (
        org_only.groupby("시설구분")["면적대비 에너지 사용비율(%)"]
        .mean()
        .to_dict()
    )

    corp_row = {
        "구 분": "전 체",
        "에너지 사용량(현재 기준)": total_cur,
        "전년대비 증감률(%)": change_vs_prev,
        "3개년 평균 에너지 사용량 대비 증감률(%)": change_vs_3y,
        "의료시설 면적대비 평균 에너지 사용비율(%)": type_avg.get("의료시설", np.nan),
        "복지시설 면적대비 평균 에너지 사용비율(%)": type_avg.get("복지시설", np.nan),
        "기타시설 면적대비 평균 에너지 사용비율(%)": type_avg.get("기타시설", np.nan),
    }

    return pd.DataFrame([corp_row])


def compute_trend_series(all_data: pd.DataFrame, org_filter: str = "공단 전체") -> pd.Series:
    """
    연도별 에너지 사용량 추이 (막대그래프용 series)
    - org_filter == "공단 전체" 이면 전체 합
    - 특정 기관명이면 해당 기관만 합
    """
    if all_data.empty:
        return pd.Series(dtype=float)

    df = add_classification_columns(all_data)

    if org_filter != "공단 전체":
        df = df[df["소속기관명"] == org_filter]

    s = df.groupby("기준연도")["연단위"].sum().sort_index()
    return s


def compute_detail_table(
    all_data: pd.DataFrame,
    base_year: int,
    org_filter: str = "공단 전체",
) -> pd.DataFrame:
    """
    '에너지 사용량 관리 대상 상세'용 상세 테이블.
    기관 + 연료 단위별로 현재·전년·3개년 평균, 면적대비 비율 등을 계산.
    """

    if all_data.empty:
        return pd.DataFrame()

    df = add_classification_columns(all_data)
    df_cur = df[df["기준연도"] == base_year].copy()
    df_prev = df[df["기준연도"] == base_year - 1].copy()
    df_3y = df[df["기준연도"].between(base_year - 3, base_year - 1)].copy()

    if org_filter != "공단 전체":
        df_cur = df_cur[df_cur["소속기관명"] == org_filter]
        df_prev = df_prev[df_prev["소속기관명"] == org_filter]
        df_3y = df_3y[df_3y["소속기관명"] == org_filter]

    if df_cur.empty:
        return pd.DataFrame()

    group_cols = ["소속기관명", "시설구분_대분류", "연료", "단위"]

    def agg_for_year(d: pd.DataFrame, year: int | None) -> pd.DataFrame:
        if d.empty:
            return pd.DataFrame(columns=group_cols + ["연면적", "에너지사용량"])
        g = (
            d.groupby(group_cols, dropna=False)
            .apply(
                lambda g_: pd.Series(
                    {
                        "연면적": g_["연면적"].dropna().iloc[0]
                        if not g_["연면적"].dropna().empty
                        else np.nan,
                        "에너지사용량": g_["연단위"].sum(),
                    }
                )
            )
            .reset_index()
        )
        if year is not None:
            g["기준연도"] = year
        return g

    cur = agg_for_year(df_cur, base_year)
    prev = agg_for_year(df_prev, base_year - 1)
    avg3 = (
        agg_for_year(df_3y, None)
        .groupby(group_cols, dropna=False)["에너지사용량"]
        .mean()
        .reset_index()
        .rename(columns={"에너지사용량": "에너지사용량_3개년평균"})
    )

    detail = cur.merge(
        prev[group_cols + ["에너지사용량"]],
        how="left",
        on=group_cols,
        suffixes=("", "_전년"),
    ).merge(
        avg3,
        how="left",
        on=group_cols,
    )

    # 증감률 계산
    detail["전년대비 증감률(%)"] = np.where(
        detail["에너지사용량_전년"] > 0,
        (detail["에너지사용량"] - detail["에너지사용량_전년"])
        / detail["에너지사용량_전년"]
        * 100,
        np.nan,
    )

    detail["3개년 평균 대비 증감률(%)"] = np.where(
        detail["에너지사용량_3개년평균"] > 0,
        (detail["에너지사용량"] - detail["에너지사용량_3개년평균"])
        / detail["에너지사용량_3개년평균"]
        * 100,
        np.nan,
    )

    detail["면적대비 에너지 사용비율(%)"] = (
        detail["연면적"] / detail["에너지사용량"] * 100
    )

    # 보기 좋게 정렬
    detail = detail[
        [
            "소속기관명",
            "시설구분_대분류",
            "연료",
            "단위",
            "연면적",
            "에너지사용량",
            "에너지사용량_전년",
            "에너지사용량_3개년평균",
            "전년대비 증감률(%)",
            "3개년 평균 대비 증감률(%)",
            "면적대비 에너지 사용비율(%)",
        ]
    ].sort_values(["소속기관명", "연료"])

    return detail


# ============================================================
# 3. 사이드바 – 업로드 & 필터
# ============================================================

st.sidebar.header("① 데이터 업로드 및 옵션")

uploaded_2021 = st.sidebar.file_uploader("2021년 백데이터 업로드", type=["xlsx", "xls"], key="f2021")
uploaded_2022 = st.sidebar.file_uploader("2022년 백데이터 업로드", type=["xlsx", "xls"], key="f2022")
uploaded_2023 = st.sidebar.file_uploader("2023년 백데이터 업로드", type=["xlsx", "xls"], key="f2023")
uploaded_2024 = st.sidebar.file_uploader("2024년 백데이터 업로드", type=["xlsx", "xls"], key="f2024")

dfs_by_year: Dict[int, pd.DataFrame] = {}

for year, up in [(2021, uploaded_2021), (2022, uploaded_2022),
                 (2023, uploaded_2023), (2024, uploaded_2024)]:
    if up is not None:
        dfs_by_year[year] = load_raw_from_excel(up, year)

if not dfs_by_year:
    st.info("좌측에서 2021–2024년 백데이터 엑셀을 업로드하면 분석이 시작됩니다.")
    st.stop()

all_data = concat_years(dfs_by_year)

available_years = sorted(dfs_by_year.keys())
default_base_year = max(available_years)

base_year = st.sidebar.selectbox(
    "기준연도 선택",
    options=available_years,
    index=available_years.index(default_base_year),
)

# 기관 목록
orgs_all = sorted(
    all_data["소속기관명"].dropna().unique().tolist()
)
org_filter = st.sidebar.selectbox(
    "소속기구 선택",
    options=["공단 전체"] + orgs_all,
    index=0,
)

# ============================================================
# 4. 본문 화면 구성
# ============================================================

st.markdown(f"### 에너지 사용량 분석  (기준연도: **{base_year}년**)")

# ------------------------------------------------------------
# 4-1. 연도별 에너지 사용량 추이 (막대그래프)
# ------------------------------------------------------------
st.subheader("1. 연도별 에너지 사용량 추이")

trend = compute_trend_series(all_data, org_filter=org_filter)
if trend.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
else:
    st.bar_chart(trend, use_container_width=True)

# ------------------------------------------------------------
# 4-2. 에너지 사용량 분석 – 공단 전체 기준 & 소속기구별
# ------------------------------------------------------------

st.subheader("2. 에너지 사용량 분석")

# (1) 소속기구별 전체 표 계산 (필터 적용 전 기준)
org_table_full = compute_org_level_table(all_data, base_year=base_year)

if org_table_full.empty:
    st.warning("해당 기준연도의 데이터가 없습니다.")
    st.stop()

# (2) 공단 전체 기준 테이블
corp_summary = compute_corp_summary(org_table_full, all_data, base_year=base_year)

with st.expander("1) 공단 전체기준", expanded=True):
    st.dataframe(
        corp_summary.style.format(
            {
                "에너지 사용량(현재 기준)": "{:,.0f}",
                "전년대비 증감률(%)": "{:+.2f}%",
                "3개년 평균 에너지 사용량 대비 증감률(%)": "{:+.2f}%",
                "의료시설 면적대비 평균 에너지 사용비율(%)": "{:.2f}%",
                "복지시설 면적대비 평균 에너지 사용비율(%)": "{:.2f}%",
                "기타시설 면적대비 평균 에너지 사용비율(%)": "{:.2f}%",
            }
        ),
        use_container_width=True,
        height=90,
    )
    st.caption("※ 시설구분별 면적대비 평균 에너지 사용비율은 각 시설구분에 속하는 기관들의 "
               "`연면적 / 에너지 사용량 * 100` 값을 단순 평균한 값입니다.")

# (3) 소속기구별 표 – 여기서만 기관 필터 적용
with st.expander("2) 소속기구별", expanded=True):
    if org_filter != "공단 전체":
        org_table_view = org_table_full[
            (org_table_full["구 분"] == org_filter) | (org_table_full["구 분"] == "합 계")
        ].reset_index(drop=True)
    else:
        org_table_view = org_table_full.copy()

    st.dataframe(
        org_table_view.style.format(
            {
                "연면적": "{:,.2f}",
                "에너지 사용량": "{:,.0f}",
                "면적대비 에너지 사용비율(%)": "{:.2f}%",
                "에너지 사용 비중(%)": "{:.2f}%",
                "3개년 평균 에너지 사용량 대비 증감률(%)": "{:+.2f}%",
                "시설별 평균 면적 대비 에너지 사용비율(%)": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

# ------------------------------------------------------------
# 4-3. 에너지 사용량 관리 대상 상세
# ------------------------------------------------------------

st.subheader("3. 에너지 사용량 관리 대상 상세")

detail_table = compute_detail_table(all_data, base_year=base_year, org_filter=org_filter)

if detail_table.empty:
    st.info("상세 데이터를 만들 수 없습니다. (해당 조건의 데이터가 없음)")
else:
    st.dataframe(
        detail_table.style.format(
            {
                "연면적": "{:,.2f}",
                "에너지사용량": "{:,.0f}",
                "에너지사용량_전년": "{:,.0f}",
                "에너지사용량_3개년평균": "{:,.0f}",
                "전년대비 증감률(%)": "{:+.2f}%",
                "3개년 평균 대비 증감률(%)": "{:+.2f}%",
                "면적대비 에너지 사용비율(%)": "{:.2f}%",
            }
        ),
        use_container_width=True,
        height=400,
    )

# ------------------------------------------------------------
# 4-4. 서술식 피드백 (간단 버전)
# ------------------------------------------------------------

st.subheader("4. 에너지 사용량 서술형 피드백")

# 간단한 규칙 기반 피드백 예시 (엑셀 formulas.json 규칙을 대체)
org_table_for_feedback = org_table_full[org_table_full["구 분"] != "합 계"].copy()

high_usage = org_table_for_feedback.sort_values(
    "시설별 평균 면적 대비 에너지 사용비율(%)", ascending=False
).head(3)

low_usage = org_table_for_feedback.sort_values(
    "시설별 평균 면적 대비 에너지 사용비율(%)", ascending=True
).head(3)

st.markdown("**① 면적당 에너지 사용이 높은 기관(상위 3개)**")
for _, r in high_usage.iterrows():
    st.write(
        f"- {r['구 분']}: 시설구분 {r['시설구분']} / "
        f"면적대비 사용비율 {r['면적대비 에너지 사용비율(%)']:.2f}% "
        f"({r['시설별 평균 면적 대비 에너지 사용비율(%)']:.1f}% of type avg)"
    )

st.markdown("**② 면적당 에너지 사용이 낮은 기관(하위 3개)**")
for _, r in low_usage.iterrows():
    st.write(
        f"- {r['구 분']}: 시설구분 {r['시설구분']} / "
        f"면적대비 사용비율 {r['면적대비 에너지 사용비율(%)']:.2f}% "
        f"({r['시설별 평균 면적 대비 에너지 사용비율(%)']:.1f}% of type avg)"
    )

st.caption(
    "※ 서술형 피드백은 현재 간단한 규칙 기반으로 작성되어 있습니다. "
    "기존 formulas.json / rules_template.json 규칙과 연결하려면, "
    "위에서 계산된 지표들을 해당 규칙 엔진에 넘겨서 문장을 생성하면 됩니다."
)
