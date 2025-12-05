import os
import streamlit as st
import pandas as pd

from modules.loader import load_all_years
from modules.analyzer import (
    build_sheet1_tables,
    compute_overall_sheet2,
    compute_facility_sheet2,
    compute_overall_feedback,
    compute_facility_feedback,
)

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="공단 에너지 사용량 · 온실가스 관리 대시보드",
    layout="wide",
)

st.title("공단 에너지 사용량 · 온실가스 관리 대시보드")

TABS = ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
tab = st.sidebar.radio("메뉴", TABS)

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------------------
# 공통: 연도별 로딩 함수
# ------------------------------------------------------------
def get_year_to_raw():
    year_to_raw, errors = load_all_years(UPLOAD_DIR)

    if errors:
        with st.expander("⚠ 파일 로딩 관련 경고/오류 보기", expanded=False):
            for msg in errors:
                st.write("•", msg)

    return year_to_raw


# ============================================================
# 📂 (1) 에너지 사용량 파일 업로드 탭
# ============================================================
if tab == "📂 에너지 사용량 파일 업로드":
    st.header("에너지 사용량 파일 업로드")

    uploaded_files = st.file_uploader(
        "《에너지 사용량관리.xlsx》 형식의 파일을 연도별로 업로드해 주세요.",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
        st.success("파일 업로드 및 저장이 완료되었습니다.")

    # 저장된 파일 목록
    st.subheader("저장된 파일 목록")
    files = sorted([fn for fn in os.listdir(UPLOAD_DIR) if fn.lower().endswith(".xlsx")])
    if not files:
        st.info("아직 업로드된 파일이 없습니다.")
    else:
        df_files = pd.DataFrame(
            [{"No": i + 1, "파일명": fn} for i, fn in enumerate(files)]
        )
        st.dataframe(df_files, use_container_width=True)

    st.divider()
    st.subheader("📘 백데이터 분석 (시트1 구조)")

    year_to_raw = get_year_to_raw()

    if not year_to_raw:
        st.info("분석 가능한 연도 데이터가 없습니다. 먼저 파일을 업로드해 주세요.")
    else:
        df_u, df_area, df_three = build_sheet1_tables(year_to_raw)

        if df_u is not None:
            st.markdown("### ① 연도 × 기관 에너지 사용량(U)")
            st.dataframe(df_u, use_container_width=True)

        if df_area is not None:
            st.markdown("### ② 연도 × 기관 연면적")
            st.dataframe(df_area, use_container_width=True)

        if df_three is not None:
            st.markdown("### ③ 연도별 3개년 평균 에너지 사용량")
            st.dataframe(df_three, use_container_width=True)


# ============================================================
# 📊 (2) 대시보드 탭
# ============================================================
elif tab == "📊 대시보드":
    year_to_raw = get_year_to_raw()

    if not year_to_raw:
        st.warning("⚠ 분석 가능한 연도 데이터가 없습니다. 먼저 파일을 업로드해 주세요.")
    else:
        years = sorted(year_to_raw.keys())
        target_year = st.sidebar.selectbox("이행연도 선택", years, index=len(years) - 1)

        # ----------------------------------------------------
        # 🔍 소속기구 필터
        # ----------------------------------------------------
        df_target = year_to_raw[target_year]

        org_list = sorted(df_target["기관명"].unique())
        selected_orgs = st.sidebar.multiselect(
            "소속기구 선택",
            options=org_list,
            default=org_list,
        )

        # 선택된 소속기구만 남긴 year_to_raw 생성
        filtered_year_to_raw = {}
        for y, df in year_to_raw.items():
            df_y = df.copy()
            if selected_orgs:
                df_y = df_y[df_y["기관명"].isin(selected_orgs)]
            filtered_year_to_raw[y] = df_y

        
        # ----------------------------------------------------
        # 상단: 에너지 사용량 추이 (레이아웃 유지)
        # ----------------------------------------------------
        st.header("에너지 사용량 추이")

        col_trend1, col_trend2 = st.columns(2)

        # (좌) 월별 에너지 사용량 추이
        with col_trend1:
            st.subheader("월별 에너지 사용량 추이")

            df_year = filtered_year_to_raw[target_year]

            monthly_chart_drawn = False

            # 월 정보가 '월' 컬럼에 있는 경우 (예: 1~12)
            if "월" in df_year.columns:
                monthly = (
                    df_year.groupby("월")["U"].sum().reset_index().sort_values("월")
                )
                monthly = monthly.set_index("월")
                st.line_chart(monthly)
                monthly_chart_drawn = True
            # '사용년월' 형태(예: 2024-01)인 경우
            elif "사용년월" in df_year.columns:
                tmp = df_year.copy()
                tmp["월"] = tmp["사용년월"].astype(str).str[-2:].astype(int)
                monthly = tmp.groupby("월")["U"].sum().reset_index().sort_values("월")
                monthly = monthly.set_index("월")
                st.line_chart(monthly)
                monthly_chart_drawn = True

            if not monthly_chart_drawn:
                st.info("월별 사용량 추이를 계산할 수 있는 '월' 또는 '사용년월' 컬럼이 없습니다.")

        # (우) 연도별 에너지 사용량 추이 (최대 5개년)
        with col_trend2:
            st.subheader("연도별 에너지 사용량 추이 (최대 5개년)")

            data_year = []
            for y in years[-5:]:
                total_u = filtered_year_to_raw[y]["U"].sum()
                data_year.append({"연도": y, "에너지사용량": total_u})

            df_trend_year = pd.DataFrame(data_year).set_index("연도")
            st.line_chart(df_trend_year)

        st.divider()

        # ----------------------------------------------------
        # 시트2: 에너지 사용량 분석
        # ----------------------------------------------------
        st.header("에너지 사용량 분석 (시트2)")

        col2_1, col2_2 = st.columns([1.4, 2.0])

        # (좌) 공단 전체 기준
        with col2_1:
            st.markdown("### 📌 공단 전체 기준 (시트2 상단)")

            overall = compute_overall_sheet2(target_year, filtered_year_to_raw)
            if overall is None:
                st.error("공단 전체 기준 분석을 계산하지 못했습니다.")
            else:
                df_overall = pd.DataFrame(
                    {
                        "항목": [
                            "에너지 사용량(현재 기준, U 합계)",
                            "전체 면적당 온실가스 배출량(V)",
                            "3개년 평균 에너지 사용량 대비 증감률",
                        ],
                        "값": [
                            overall["에너지사용량"],
                            overall["전체면적당온실가스"],
                            overall["3개년평균대비증감"],
                        ],
                    }
                )
                st.dataframe(df_overall, use_container_width=True)

                # 시설구분별 평균 W
                st.markdown("#### 시설구분별 평균 에너지 사용량(W)")
                df_w = pd.DataFrame(
                    [
                        {"시설구분": k, "평균W": v}
                        for k, v in overall["시설구분평균W"].items()
                    ]
                )
                st.dataframe(df_w, use_container_width=True)

        # (우) 소속기구별 분석
        with col2_2:
            st.markdown("### 🏢 소속기구별 분석 (시트2 하단)")

            df_fac = compute_facility_sheet2(target_year, filtered_year_to_raw)
            if df_fac is None or df_fac.empty:
                st.error("소속기구별 분석 표를 생성하지 못했습니다.")
            else:
                st.dataframe(df_fac, use_container_width=True)

        st.divider()

        # ----------------------------------------------------
        # 시트3: 피드백
        # ----------------------------------------------------
        st.header("피드백 (시트3)")

        # (상단) 공단 전체 피드백
        st.markdown("### 📌 공단 전체 피드백 (시트3 상단)")

        fb_overall = compute_overall_feedback(target_year, filtered_year_to_raw)
        if fb_overall is None:
            st.error("공단 전체 피드백을 계산하지 못했습니다.")
        else:
            df_fb_overall = pd.DataFrame(
                {
                    "항목": ["권장 에너지 사용량", "전년대비 감축률", "3개년 평균 대비 감축률"],
                    "값": [
                        fb_overall["권장사용량"],
                        fb_overall["전년대비감축률"],
                        fb_overall["3개년평균감축률"],
                    ],
                }
            )
            st.dataframe(df_fb_overall, use_container_width=True)

        # (하단) 소속기구별 피드백
        st.markdown("### 🏢 소속기구별 피드백 (시트3 하단)")

        fb_fac1, fb_fac2 = compute_facility_feedback(target_year, filtered_year_to_raw)

        if fb_fac1 is None or fb_fac2 is None:
            st.error("소속기구별 피드백 표를 계산하지 못했습니다.")
        else:
            st.markdown("#### ① 기관별 피드백 요약")
            st.dataframe(fb_fac1, use_container_width=True)

            st.markdown("#### ② 관리대상(O/X) 상세")
            st.dataframe(fb_fac2, use_container_width=True)


# ============================================================
# 🔧 (3) 디버그 / 진단 탭
# ============================================================
elif tab == "🔧 디버그 / 진단":
    st.header("디버그 / 진단")
    st.write("• 로딩된 연도 / df_raw 구조 확인")

    year_to_raw = get_year_to_raw()
    st.write("로딩된 연도:", list(year_to_raw.keys()))

    if year_to_raw:
        sample_year = sorted(year_to_raw.keys())[0]
        st.subheader(f"샘플 df_raw 미리보기 ({sample_year}년)")
        st.dataframe(year_to_raw[sample_year].head(), use_container_width=True)
