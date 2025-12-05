import os
from typing import List, Dict

import pandas as pd
import streamlit as st

from modules.loader import load_all_years, load_monthly_usage
from modules.analyzer import (
    build_sheet1_tables,
    compute_overall_sheet2,
    compute_facility_sheet2,
    compute_overall_feedback,
    compute_facility_feedback,
    generate_overall_comment,
    generate_org_comments,
)

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="공단 에너지 사용량 · 온실가스 관리 대시보드",
    layout="wide",
)

st.title("공단 에너지 사용량 · 온실가스 관리 대시보드")

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------------------
# 연도별 데이터 공통 로딩 (한 번만)
# ------------------------------------------------------------
year_to_raw, load_errors = load_all_years(UPLOAD_DIR)

# 사이드바에 로딩 오류 표시
if load_errors:
    with st.sidebar.expander("⚠ 파일 로딩 관련 경고/오류 보기", expanded=False):
        for msg in load_errors:
            st.write("•", msg)


# ------------------------------------------------------------
# 📌 사이드바 필터 – 요청하신 레이아웃
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 필터")

    # 1) 보기 범위: 공단 전체 / 기관별
    view_scope = st.radio("보기 범위", ["공단 전체", "기관별"], index=0)

    # 2) 이행연도 선택
    if year_to_raw:
        years = sorted(year_to_raw.keys())
        default_year_idx = len(years) - 1  # 최신 연도 기본 선택
        selected_year = st.selectbox("이행연도 선택", years, index=default_year_idx)
    else:
        selected_year = None
        st.selectbox("이행연도 선택", ["(데이터 없음)"], index=0)

    # 3) 기관 선택 (기관별 선택 시에만)
    if selected_year is not None and year_to_raw:
        df_for_org = year_to_raw[selected_year]
        org_list = sorted(df_for_org["기관명"].astype(str).unique())
    else:
        org_list = []

    if view_scope == "기관별":
        # MultiSelect로 구현하지만, 기본은 하나만 선택된 상태라
        # UI 상으로는 스샷과 비슷하게 동작
        selected_orgs: List[str] = st.multiselect(
            "기관 선택",
            options=org_list,
            default=org_list[:1] if org_list else [],
        )
    else:
        # 공단 전체일 때는 모든 기관 사용
        selected_orgs = org_list

    st.markdown("### 에너지 종류 필터 (추후 확장용)")
    energy_type = st.selectbox(
        "에너지 종류",
        options=["전체"],
        index=0,
        help="현재는 '전체'만 제공되며, 향후 전기/가스 등으로 확장 예정입니다.",
    )

# ------------------------------------------------------------
# 상단 탭 메뉴 – 대시보드 / 업로드 / 디버그
# (요청하신 것처럼 제목 아래에 가로 메뉴 배치)
# ------------------------------------------------------------
tab_dashboard, tab_upload, tab_debug = st.tabs(
    ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
)


# ============================================================
# 📊 (1) 대시보드 탭
# ============================================================
with tab_dashboard:
    if not year_to_raw or selected_year is None:
        st.warning("⚠ 분석 가능한 연도 데이터가 없습니다. 먼저 파일을 업로드해 주세요.")
    else:
        # 선택된 범위(공단 전체 / 특정 기관들)에 맞춰 df를 필터링한 year_to_raw 생성
        filtered_year_to_raw: Dict[int, pd.DataFrame] = {}
        for y, df in year_to_raw.items():
            df_y = df.copy()
            if selected_orgs:
                df_y = df_y[df_y["기관명"].astype(str).isin(selected_orgs)]
            filtered_year_to_raw[y] = df_y

        st.subheader("에너지 사용량 추이")

        col_trend1, col_trend2 = st.columns(2)

        # (좌) 월별 에너지 사용량 추이
        with col_trend1:
            st.markdown("##### 월별 에너지 사용량 추이")
            monthly_df = load_monthly_usage(UPLOAD_DIR, selected_year, selected_orgs)

            if monthly_df is not None:
                st.line_chart(monthly_df)
            else:
                st.info("월별 사용량 추이를 계산할 수 있는 컬럼이 원본 파일에 없습니다.")

        # (우) 연도별 에너지 사용량 추이 (최대 5개년)
        with col_trend2:
            st.markdown("##### 연도별 에너지 사용량 추이 (최대 5개년)")
            years_sorted = sorted(filtered_year_to_raw.keys())
            data_year = []
            for y in years_sorted[-5:]:
                total_u = filtered_year_to_raw[y]["U"].sum()
                data_year.append({"연도": y, "에너지사용량": total_u})

            if data_year:
                df_trend_year = pd.DataFrame(data_year).set_index("연도")
                st.line_chart(df_trend_year)
            else:
                st.info("연도별 에너지 사용량 추이를 계산할 수 있는 데이터가 없습니다.")

        st.divider()

        # -----------------------------
        # 시트2: 에너지 사용량 분석
        # -----------------------------
        st.subheader("에너지 사용량 분석 (시트2)")

        col2_1, col2_2 = st.columns([1.4, 2.0])

        # (좌) 공단 전체 기준 (상단 표)
        with col2_1:
            st.markdown("###### 📌 공단 전체 기준 (시트2 상단)")
            overall = compute_overall_sheet2(selected_year, filtered_year_to_raw)
            if overall is None:
                st.error("공단 전체 기준 분석을 계산하지 못했습니다.")
            else:
                df_overall = pd.DataFrame(
                    [
                        {
                            "에너지 사용량(현재 기준)": overall["에너지사용량"],
                            "전년 대비 증감률": overall["전년대비증감률"],
                            "3개년 평균 대비 증감률": overall["3개년평균대비증감률"],
                            "의료시설 평균W": overall["의료시설평균W"],
                            "복지시설 평균W": overall["복지시설평균W"],
                            "기타시설 평균W": overall["기타시설평균W"],
                        }
                    ],
                    index=["공단 전체"],
                )
                st.dataframe(df_overall, use_container_width=True)

        # (우) 소속기구별 분석 (하단 표)
        with col2_2:
            st.markdown("###### 🏢 소속기구별 분석 (시트2 하단)")
            df_fac = compute_facility_sheet2(selected_year, filtered_year_to_raw)
            if df_fac is None or df_fac.empty:
                st.error("소속기구별 분석 표를 생성하지 못했습니다.")
            else:
                st.dataframe(df_fac, use_container_width=True)

        st.divider()

        # -----------------------------
        # 시트3: 피드백
        # -----------------------------
        st.subheader("피드백 (시트3)")

        # (상단) 공단 전체 피드백
        st.markdown("###### 📌 공단 전체 피드백 (시트3 상단)")
        fb_overall = compute_overall_feedback(selected_year, filtered_year_to_raw)
        if fb_overall is None:
            st.error("공단 전체 피드백을 계산하지 못했습니다.")
        else:
            df_fb_overall = pd.DataFrame(
                [
                    {
                        "권장 사용량": fb_overall["권장사용량"],
                        "전년 대비 감축률": fb_overall["전년대비감축률"],
                        "3개년 평균 대비 감축률": fb_overall["3개년평균감축률"],
                    }
                ],
                index=["공단 전체"],
            )
            st.dataframe(df_fb_overall, use_container_width=True)

        # (하단) 소속기구별 피드백 2개 표
        st.markdown("###### 🏢 소속기구별 피드백 (시트3 하단)")
        fb_fac1, fb_fac2 = compute_facility_feedback(selected_year, filtered_year_to_raw)

        if fb_fac1 is None or fb_fac2 is None:
            st.error("소속기구별 피드백 표를 계산하지 못했습니다.")
        else:
            st.markdown("**① 기관별 피드백 요약**")
            st.dataframe(fb_fac1, use_container_width=True)

            st.markdown("**② 관리대상(O/X) 상세**")
            st.dataframe(fb_fac2, use_container_width=True)

            # 서술형 코멘트
            st.markdown("### 📝 AI 기반 요약 코멘트")
            overall_comment = generate_overall_comment(selected_year, overall, df_fac)
            st.markdown(f"**공단 전체 요약**  \n{overall_comment}")

            org_comments = generate_org_comments(fb_fac1, fb_fac2)
            with st.expander("기관별 상세 코멘트 보기", expanded=False):
                for txt in org_comments:
                    st.markdown(f"- {txt}")


# ============================================================
# 📂 (2) 에너지 사용량 파일 업로드 탭
# ============================================================
with tab_upload:
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
        st.success("파일 업로드 및 저장이 완료되었습니다. 화면을 새로고침하면 분석에 반영됩니다.")

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
# 🔧 (3) 디버그 / 진단 탭
# ============================================================
with tab_debug:
    st.header("디버그 / 진단")
    st.write("• 로딩된 연도 / df_raw 구조 확인 및 매핑 점검용 화면입니다.")

    st.subheader("로딩된 연도 목록")
    st.write(sorted(year_to_raw.keys()))

    if year_to_raw:
        dbg_year = st.selectbox(
            "미리보기 연도 선택",
            options=sorted(year_to_raw.keys()),
        )
        st.markdown("#### df_raw 미리보기")
        st.dataframe(year_to_raw[dbg_year].head(), use_container_width=True)

        st.markdown("#### df_raw 컬럼 목록")
        st.write(list(year_to_raw[dbg_year].columns))
    else:
        st.info("현재 로딩된 df_raw 데이터가 없습니다.")
