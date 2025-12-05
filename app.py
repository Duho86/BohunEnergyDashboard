import streamlit as st
import pandas as pd
import os

from modules.loader import load_all_years
from modules.analyzer import (
    build_sheet1_tables,
    compute_overall_sheet2,
    compute_facility_sheet2,
    compute_overall_feedback,
    compute_facility_feedback,
)

# ==============================================
# 기본 UI 설정
# ==============================================
st.set_page_config(
    page_title="공단 에너지 사용량 · 온실가스 관리 대시보드",
    layout="wide",
)

st.title("공단 에너지 사용량 · 온실가스 관리 대시보드")

TABS = ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
tab = st.sidebar.radio("메뉴", TABS)

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==============================================================
# 📂 (1) 업로드 화면
# ==============================================================
if tab == "📂 에너지 사용량 파일 업로드":
    st.header("에너지 사용량 파일 업로드")

    uploaded = st.file_uploader(
        "에너지 사용량관리 .xlsx 파일 업로드",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    # 저장
    if uploaded:
        for f in uploaded:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
        st.success("파일 업로드 및 저장 완료.")

    # 저장된 파일 목록 표시
    st.subheader("저장된 파일 목록")
    files = sorted(os.listdir(UPLOAD_DIR))
    if len(files) == 0:
        st.info("아직 업로드된 파일이 없습니다.")
    else:
        df_files = []
        for i, fn in enumerate(files):
            df_files.append({"번호": i, "파일명": fn})
        st.dataframe(pd.DataFrame(df_files), use_container_width=True)

    # ----------------------------
    # 시트1 백데이터 분석 표 출력
    # ----------------------------
    st.divider()
    st.subheader("📘 백데이터 분석 (시트1)")

    year_to_raw = load_all_years(UPLOAD_DIR)

    if len(year_to_raw) == 0:
        st.info("분석 가능한 연도 데이터가 없습니다.")
    else:
        df_u, df_area, df_three = build_sheet1_tables(year_to_raw)

        if df_u is not None:
            st.markdown("### ① 연도 × 기관 에너지 사용량(U)")
            st.dataframe(df_u, use_container_width=True)

        if df_area is not None:
            st.markdown("### ② 연도 × 기관 연면적")
            st.dataframe(df_area, use_container_width=True)

        if df_three is not None:
            st.markdown("### ③ 연도별 3개년 평균 대비 분석")
            st.dataframe(df_three, use_container_width=True)


# ==============================================================
# 📊 (2) 대시보드 화면
# ==============================================================
elif tab == "📊 대시보드":
    year_to_raw = load_all_years(UPLOAD_DIR)

    if len(year_to_raw) == 0:
        st.warning("⚠ 분석 가능한 연도 데이터가 없습니다. 먼저 파일을 업로드하세요.")
        st.stop()

    years = sorted(year_to_raw.keys())
    target_year = st.sidebar.selectbox("이행연도 선택", years, index=len(years) - 1)

    # ----------------------------------------------------------
    # 에너지 사용량 추이 (간단 버전: 연도별 합계 추이)
    # ----------------------------------------------------------
    st.header("에너지 사용량 추이")

    # 연도별 에너지 사용량 합계
    yearly_totals = []
    for y, df_y in year_to_raw.items():
        yearly_totals.append({"연도": y, "에너지사용량": df_y["에너지사용량"].sum()})

    df_yearly = pd.DataFrame(yearly_totals).sort_values("연도")

    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        st.subheader("연도별 에너지 사용량 추이")
        st.line_chart(df_yearly.set_index("연도"))

    with col_trend2:
        st.subheader("연도별 에너지 사용량 (막대)")
        st.bar_chart(df_yearly.set_index("연도"))

    st.divider()

    # ----------------------------------------------------------
    # 시트2 — 에너지 사용량 분석
    # ----------------------------------------------------------
    st.header("에너지 사용량 분석 (시트2)")

    col1, col2 = st.columns([2, 3])

    # -----------------------------
    # (시트2 상단) 공단 전체 분석
    # -----------------------------
    st.write("디버그 - year_to_raw keys:", list(year_to_raw.keys()))
    st.write("디버그 - 선택 연도:", target_year)

    overall = compute_overall_sheet2(target_year, year_to_raw)

    if overall is None:
        st.error("데이터 문제로 분석이 불가합니다.")
        st.stop()

    with col1:
        st.markdown("### 📌 공단 전체 기준 (시트2 상단)")

        df_overall = pd.DataFrame(
            {
                "항목": [
                    "에너지 사용량(현재 기준)",
                    "전년대비 증감률",
                    "3개년 평균 에너지 사용량 대비 증감률",
                ],
                "값": [
                    overall["에너지사용량"],
                    overall["전년대비증감"],
                    overall["3개년평균대비증감"],
                ],
            }
        )

        st.dataframe(df_overall, use_container_width=True)

        # 시설군 평균 W
        st.markdown("#### 시설구분별 면적대비 평균 에너지 사용비율(W)")
        st.dataframe(
            pd.DataFrame(
                overall["시설구분평균"].items(), columns=["시설구분", "평균비율"]
            ),
            use_container_width=True,
        )

    # ---------------------------------
    # (시트2 하단) 소속기구별 분석 표
    # ---------------------------------
    with col2:
        st.markdown("### 🏢 소속기구별 분석 (시트2 하단)")
        df_fac = compute_facility_sheet2(target_year, year_to_raw)

        if df_fac is None:
            st.error("소속기구별 분석을 생성할 수 없습니다.")
        else:
            df_out = df_fac[
                [
                    "기관명",
                    "시설구분",
                    "연면적",
                    "에너지사용량",
                    "면적대비에너지비율",
                    "에너지비중",
                    "3개년평균대비증감률",
                    "시설군평균대비비율",
                ]
            ]
            st.dataframe(df_out, use_container_width=True)

    st.divider()

    # ----------------------------------------------------------
    # 시트3 — 피드백
    # ----------------------------------------------------------
    st.header("피드백 (시트3)")

    st.write("디버그 - year_to_raw keys:", list(year_to_raw.keys()))
    st.write("디버그 - 선택 연도:", target_year)

    # -----------------------
    # (시트3 상단) 공단 전체 피드백
    # -----------------------
    st.markdown("### 📌 공단 전체 기준 (시트3 상단)")

    fb_all = compute_overall_feedback(target_year, year_to_raw)

    if fb_all is None:
        st.error(
            "공단 전체 피드백(시트3 상단)을 계산하지 못했습니다. "
            "year_to_raw 또는 target_year 데이터를 확인하세요."
        )
    else:
        df_fb_all = pd.DataFrame(
            {
                "항목": ["권장 에너지 사용량", "전년대비 감축률", "3개년 대비 감축률"],
                "값": [
                    fb_all.get("권장사용량"),
                    fb_all.get("전년대비감축률"),
                    fb_all.get("3개년평균감축률"),
                ],
            }
        )
        st.dataframe(df_fb_all, use_container_width=True)

    # -----------------------
    # (시트3 하단) 소속기구별 피드백 2개 표
    # -----------------------
    st.markdown("### 🏢 소속기구별 피드백 (시트3 하단)")

    fb_facility = compute_facility_feedback(target_year, year_to_raw)

    if (
        fb_facility is None
        or fb_facility[0] is None
        or fb_facility[1] is None
    ):
        st.error(
            "소속기구별 피드백(시트3 하단)을 계산하지 못했습니다. "
            "analyzer.compute_facility_feedback 로직과 연도별 데이터를 확인하세요."
        )
    else:
        df_fb1, df_fb2 = fb_facility

        st.markdown("#### ① 기관별 피드백 요약")
        st.dataframe(df_fb1, use_container_width=True)

        st.markdown("#### ② 에너지 사용량 관리대상(O/X) 상세")
        st.dataframe(df_fb2, use_container_width=True)


# ==============================================================
# 🔧 (3) 디버그 탭
# ==============================================================
elif tab == "🔧 디버그 / 진단":
    st.header("디버그 / 진단")

    st.write("• 로딩된 연도 / df_raw 구조 확인")
    year_to_raw = load_all_years(UPLOAD_DIR)

    st.json({"로딩된연도": list(year_to_raw.keys())})

    if len(year_to_raw) > 0:
        y = list(year_to_raw.keys())[0]
        st.write(f"샘플 연도 {y} df_raw 미리보기")
        st.dataframe(year_to_raw[y].head(), use_container_width=True)
