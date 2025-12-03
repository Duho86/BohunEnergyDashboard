# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

import pandas as pd
import streamlit as st

from modules import loader, analyzer, feedback, baseline as baseline_mod


# ============================
# 기본 설정
# ============================

st.set_page_config(
    page_title="공단 에너지 사용량 · 온실가스 관리 대시보드",
    layout="wide",
)

st.title("공단 에너지 사용량 · 온실가스 관리 대시보드")


DATA_DIR = Path("data")
ENERGY_DIR = DATA_DIR / "energy"
BASELINE_PATH = DATA_DIR / "baseline.json"


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    loader.ensure_energy_dir(ENERGY_DIR)


# ============================
# 데이터 로딩 헬퍼
# ============================

def load_all_energy_data(base_dir: Path = ENERGY_DIR):
    """
    data/energy/ 아래의 모든 .xlsx 파일을 표준 스키마로 로드 및 통합.
    """
    ensure_dirs()
    dfs = []
    meta_list = []
    errors = []

    for xlsx_path in sorted(base_dir.glob("*.xlsx")):
        try:
            df_std, year = loader.load_energy_xlsx(xlsx_path)
            dfs.append(df_std)

            stat = xlsx_path.stat()
            meta_list.append(
                {
                    "연도": year,
                    "파일명": xlsx_path.name,
                    "경로": str(xlsx_path),
                    "업로드시간": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
        except loader.EnergyDataError as e:
            errors.append({"파일명": xlsx_path.name, "에러": str(e)})
        except Exception as e:
            errors.append({"파일명": xlsx_path.name, "에러": f"알 수 없는 오류: {e}"})

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = None

    return df_all, meta_list, errors


def get_year_metrics(
    year: int,
    annual_total: pd.DataFrame,
    annual_total_with_baseline: pd.DataFrame,
):
    """
    선택 연도에 대한 주요 지표를 가져온다.
    """
    row_total = annual_total[annual_total["연도"] == year]
    row_base = annual_total_with_baseline[
        annual_total_with_baseline["연도"] == year
    ]

    if row_total.empty:
        actual = None
    else:
        actual = float(row_total["연간 온실가스 배출량"].sum())

    if row_base.empty:
        baseline = reduction = ratio = None
    else:
        baseline = row_base["기준배출량"].iloc[0]
        reduction = row_base["감축률(%)"].iloc[0]
        ratio = row_base["배출비율"].iloc[0]

    return actual, baseline, reduction, ratio


# ============================
# 세션 상태 초기화
# ============================

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

ensure_dirs()

# baseline.json 로딩 (사용자 입력값만 사용)
baseline_records = baseline_mod.load_baseline_records(BASELINE_PATH)
baseline_map = baseline_mod.get_baseline_map(baseline_records)

# ============================
# 탭 구성: 대시보드 / 기준배출량 관리 / 디버그
# ============================

tab_dashboard, tab_baseline, tab_debug = st.tabs(
    ["📊 대시보드", "⚙️ 기준배출량 관리", "🔧 디버그/진단"]
)

# ============================================================
# 📊 1) 대시보드 탭
# ============================================================

with tab_dashboard:
    # ------------------------------
    # 파일 업로드 및 저장된 파일 목록
    # ------------------------------
    st.markdown("### 월별 에너지 사용량 파일 업로드")

    upload_col1, upload_col2 = st.columns([1.2, 2])

    new_file_processed = False

    with upload_col1:
        uploaded_files = st.file_uploader(
            "에너지 사용량관리 .xlsx 파일 업로드",
            type=["xlsx"],
            accept_multiple_files=True,
            help="예: 2022년 에너지 사용량관리.xlsx",
        )
        st.caption("※ 업로드 시 data/energy/ 폴더에 저장되고, 대시보드가 자동 갱신됩니다.")

        if uploaded_files:
            for f in uploaded_files:
                if f.name in st.session_state["processed_uploads"]:
                    continue

                try:
                    _, year, saved_path = loader.process_uploaded_energy_file(
                        file_obj=f,
                        original_filename=f.name,
                        base_dir=ENERGY_DIR,
                    )
                    st.session_state["processed_uploads"].add(f.name)
                    st.success(f"{f.name} (연도: {year}) 업로드 및 저장 완료")
                    new_file_processed = True
                except loader.EnergyDataError as e:
                    st.error(f"{f.name} 업로드 처리 중 오류:\n{e}")
                except Exception as e:
                    st.error(f"{f.name} 업로드 처리 중 알 수 없는 오류가 발생했습니다: {e}")

        if new_file_processed:
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

    with upload_col2:
        st.markdown("#### 저장된 연도별 파일 목록")
        df_all, file_meta_list, load_errors = load_all_energy_data(ENERGY_DIR)

        if file_meta_list:
            df_files = pd.DataFrame(file_meta_list)
            df_files = df_files.sort_values(
                ["연도", "업로드시간"], ascending=[False, False]
            ).reset_index(drop=True)
            st.table(df_files[["연도", "파일명", "업로드시간"]])
        else:
            st.info("현재 data/energy/ 폴더에 저장된 파일이 없습니다.")

        if load_errors:
            with st.expander("⚠️ 로딩 오류가 발생한 파일 목록 보기"):
                st.write(pd.DataFrame(load_errors))

    st.markdown("---")

    if df_all is None or df_all.empty:
        st.warning("아직 분석할 에너지 사용량 데이터가 없습니다. 상단에서 파일을 업로드해 주세요.")
        st.stop()

    # ------------------------------
    # analyzer 집계
    # ------------------------------
    try:
        datasets = analyzer.build_dashboard_datasets(df_all, baseline_map=baseline_map)
    except Exception as e:
        st.error(f"데이터 집계 중 오류가 발생했습니다: {e}")
        with st.expander("자세한 오류 정보 보기 (개발용)"):
            st.code(traceback.format_exc())
        st.stop()

    monthly_by_agency = datasets["monthly_by_agency"]
    monthly_total = datasets["monthly_total"]
    annual_by_agency = datasets["annual_by_agency"]
    annual_total = datasets["annual_total"]
    annual_total_with_baseline = datasets["annual_total_with_baseline"]

    # ------------------------------
    # 필터 UI (사이드바)
    # ------------------------------
    st.sidebar.header("필터")

    years = sorted(df_all["연도"].unique().tolist())
    current_year = max(years) if years else None

    view_mode = st.sidebar.radio("보기 범위", ["공단 전체", "기관별"], index=0)

    agency_list = sorted(df_all["기관명"].unique().tolist())

    if view_mode == "공단 전체":
        selected_agency = None
        st.sidebar.markdown("**기관:** 공단 전체 기준")
    else:
        selected_agency = st.sidebar.selectbox("기관 선택", options=agency_list, index=0)

    selected_year = st.sidebar.selectbox(
        "이행연도 선택",
        options=years,
        index=years.index(current_year) if current_year in years else 0,
    )

    st.sidebar.markdown("**에너지 종류 필터 (추후 확장용)**")
    st.sidebar.multiselect(
        "에너지 종류",
        options=["전체", "전기", "가스", "신재생"],
        default=["전체"],
        help="현재 버전은 '전체' 기준으로만 집계됩니다.",
    )

    # ------------------------------
    # KPI 카드
    # ------------------------------
    st.markdown("### 주요 지표")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    actual_emission, baseline_emission, reduction_rate_pct, ratio_to_baseline = get_year_metrics(
        selected_year, annual_total, annual_total_with_baseline
    )

    with kpi_col1:
        st.metric("선택 연도", f"{selected_year}년")

    with kpi_col2:
        if actual_emission is not None:
            st.metric("연간 온실가스 배출량(공단)", f"{actual_emission:,.0f} tCO2eq")
        else:
            st.metric("연간 온실가스 배출량(공단)", "-")

    with kpi_col3:
        if reduction_rate_pct is not None and not pd.isna(reduction_rate_pct):
            st.metric("감축률(전체 기준)", f"{reduction_rate_pct:,.1f} %")
        else:
            st.metric("감축률(전체 기준)", "기준배출량 정보 없음")

    with kpi_col4:
        if ratio_to_baseline is not None and not pd.isna(ratio_to_baseline):
            st.metric("기준배출량 대비 배출비율", f"{ratio_to_baseline * 100:,.1f} %")
        else:
            st.metric("기준배출량 대비 배출비율", "기준배출량 정보 없음")

    # ------------------------------
    # 이행연도 월별 추이 / 최근 5개년 추이
    # ------------------------------
    left_col, right_col = st.columns([2, 1.4])

    with left_col:
        st.markdown("#### 이행연도 월별 온실가스 추이")

        if view_mode == "공단 전체":
            df_month_plot = (
                monthly_total[monthly_total["연도"] == selected_year]
                .sort_values("월")
                .set_index("월")
            )
        else:
            df_month_plot = (
                monthly_by_agency[
                    (monthly_by_agency["연도"] == selected_year)
                    & (monthly_by_agency["기관명"] == selected_agency)
                ]
                .sort_values("월")
                .set_index("월")
            )

        if df_month_plot.empty:
            st.info("선택한 조건에 해당하는 월별 데이터가 없습니다.")
        else:
            st.line_chart(df_month_plot["월별 온실가스 환산량"])

        st.caption("※ analyzer.get_monthly_ghg() 결과를 사용하여 월별 추이를 시각화합니다.")

    with right_col:
        st.markdown("#### 최근 5개년 연간 배출량 추이")

        if view_mode == "공단 전체":
            df_recent, _ = analyzer.get_recent_years_ghg(
                annual_total, n_years=5, base_year=selected_year
            )
        else:
            annual_agency = annual_by_agency[annual_by_agency["기관명"] == selected_agency]
            if annual_agency.empty:
                df_recent = pd.DataFrame()
            else:
                df_recent, _ = analyzer.get_recent_years_ghg(
                    annual_agency, n_years=5, base_year=selected_year
                )

        if df_recent.empty:
            st.info("최근 5개년에 대한 데이터가 충분하지 않습니다.")
        else:
            df_recent_plot = df_recent.sort_values("연도").set_index("연도")
            st.bar_chart(df_recent_plot["연간 온실가스 배출량"])

        st.caption("※ analyzer.get_recent_years_ghg() 결과를 이용하여 5개년 추이를 표시합니다.")

    # ------------------------------
    # 전망분석 / 피드백용 테이블 계산
    # ------------------------------
    try:
        projection_tables = analyzer.build_projection_tables(
            annual_total=annual_total,
            annual_by_agency=annual_by_agency,
            baseline_map=baseline_map,
            target_year=selected_year,
        )
    except Exception as e:
        projection_tables = None
        st.error(f"전망분석 테이블 계산 중 오류가 발생했습니다: {e}")

    try:
        feedback_tables = analyzer.build_feedback_tables(
            annual_total=annual_total,
            annual_by_agency=annual_by_agency,
            target_year=selected_year,
        )
    except Exception as e:
        feedback_tables = None
        st.error(f"피드백 테이블 계산 중 오류가 발생했습니다: {e}")

    # ------------------------------
    # 전망분석 섹션 (건물 기준 요약 + 상세)
    # ------------------------------
    st.markdown("---")
    st.markdown("### 전망분석")

    st.caption(
        "※ 엑셀 '에너지 사용량 분석.xlsx' 시트1의 2~4행(공단 전체), 7~27행(소속기구별) 구조를 "
        "참고하여 자동 생성된 요약 표입니다. 5행·28행의 설명행은 계산 규칙으로만 사용되며 화면에는 출력하지 않습니다."
    )

    # (1) 건물 기준 요약 (구분 | 값)
    baseline_for_year = baseline_map.get(selected_year)
    actual_for_year = actual_emission
    if (
        baseline_for_year is None
        or pd.isna(baseline_for_year)
        or actual_for_year is None
        or pd.isna(actual_for_year)
        or baseline_for_year == 0
    ):
        reduction_simple = pd.NA
    else:
        reduction_simple = (baseline_for_year - actual_for_year) / baseline_for_year * 100.0

    summary_rows = [
        {"구분": "기준배출량", "값": baseline_for_year},
        {"구분": "이행연도 배출량(소계)", "값": actual_for_year},
        {"구분": "감축률(소계)", "값": reduction_simple},
    ]
    st.markdown("#### 전망분석(건물 기준) 요약")
    st.table(pd.DataFrame(summary_rows))

    # (2) 시트1 구조를 반영한 상세 전망분석 표 (공단 전체 / 소속기구별)
    if projection_tables is None:
        st.info("전망분석 상세 테이블을 생성할 수 없습니다. 상단 오류 메시지를 확인해 주세요.")
    else:
        col_proj1, col_proj2 = st.columns(2)

        with col_proj1:
            st.markdown("#### 공단 전체 전망분석 (시트1 2~4행 구조)")
            st.table(projection_tables["overall"])

        with col_proj2:
            st.markdown("#### 소속기구별 전망분석 (시트1 7~27행 구조)")
            st.dataframe(projection_tables["by_agency"])

    # ------------------------------
    # 피드백 섹션 (시트2 기반 + 자연어 피드백)
    # ------------------------------
    st.markdown("---")
    st.markdown("### 피드백")

    st.caption(
        "※ 엑셀 '에너지 사용량 분석.xlsx' 시트2의 2~4행(공단 전체), 7~27행(소속기구별) 구조를 "
        "참고하여 금년/전년/5개년 추세를 요약한 표입니다. 5행·28행의 설명행은 계산 규칙으로만 사용됩니다."
    )

    if feedback_tables is None:
        st.info("피드백 테이블을 생성할 수 없습니다. 상단 오류 메시지를 확인해 주세요.")
    else:
        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            st.markdown("#### 공단 전체 피드백(표)")
            st.table(feedback_tables["overall"])

        with fb_col2:
            st.markdown("#### 소속기구별 피드백(표)")
            st.dataframe(feedback_tables["by_agency"])

    st.markdown("#### 공단 전체 분석·코멘트")

    if actual_emission is None:
        st.info("선택한 연도에 대한 연간 배출량 정보가 없어, 분석·피드백 문장을 생성할 수 없습니다.")
    else:
        recent_total_df, _ = analyzer.get_recent_years_ghg(
            annual_total,
            n_years=5,
            base_year=selected_year,
        )

        df_selected_year = df_all[df_all["연도"] == selected_year]
        if not df_selected_year.empty and "월" in df_selected_year.columns:
            current_month = int(df_selected_year["월"].max())
        else:
            current_month = None

        feedback_text = feedback.generate_overall_feedback(
            year=selected_year,
            actual_emission=actual_emission,
            baseline_emission=baseline_for_year,
            reduction_rate_pct=reduction_simple,
            ratio_to_baseline=None,  # 필요하면 계산해서 넣을 수 있음
            recent_total_df=recent_total_df,
            current_month=current_month,
        )

        st.write(feedback_text)

    # (옵션) 표준 스키마 미리보기
    with st.expander("표준 스키마 데이터 미리보기 (디버깅용)"):
        st.write(df_all.head())
        st.caption("※ loader.normalize_energy_dataframe() 결과를 concat한 전체 데이터입니다.")


# ============================================================
# ⚙️ 2) 기준배출량 관리 탭
# ============================================================

with tab_baseline:
    st.header("기준배출량 관리")

    st.caption(
        "연도별 온실가스 기준배출량(tCO2e)과 이행연도 대상 여부를 사용자 입력으로 관리합니다. "
        "여기서 입력한 값만이 전망분석·감축률 계산에 사용되며, 시스템이 자동으로 기준배출량을 산정하지 않습니다."
    )

    # 세션 상태에 baseline_records 보존
    if "baseline_records" not in st.session_state:
        st.session_state["baseline_records"] = baseline_records.copy()

    records = st.session_state["baseline_records"]

    # ----- 목록 표시 (연도 | 기준배출량 | 이행연도 대상 | [수정]) -----
    st.markdown("#### 연도별 기준배출량 목록")

    if not records:
        st.info("등록된 기준배출량이 없습니다. 아래에서 [추가]하여 입력해 주세요.")
    else:
        # 최신 연도가 위로 오도록 정렬
        for year in sorted(records.keys(), reverse=True):
            rec = records[year]
            baseline_val = rec.get("baseline")
            is_target = rec.get("is_target", False)

            c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
            c1.write(f"{year}")
            c2.write(f"{baseline_val:,.0f} tCO2eq" if baseline_val is not None else "-")
            c3.write("O" if is_target else "X")
            if c4.button("수정", key=f"edit_{year}"):
                st.session_state["baseline_edit_year"] = year

    st.markdown("---")

    # ----- 추가/수정 폼 -----
    st.markdown("#### 기준배출량 추가 / 수정")

    edit_year = st.session_state.get("baseline_edit_year", None)

    if edit_year is not None and edit_year in records:
        # 수정 모드
        default_year = int(edit_year)
        default_baseline = records[edit_year].get("baseline") or 0.0
        default_is_target = bool(records[edit_year].get("is_target", False))
        st.info(f"{default_year}년 기준배출량을 수정 중입니다.")
    else:
        # 추가 모드 (기본값: 가장 최근 연도 + 1 또는 올해)
        default_year = datetime.now().year
        if records:
            default_year = max(max(records.keys()) + 1, default_year)
        default_baseline = 0.0
        default_is_target = False

    with st.form("baseline_edit_form"):
        year_input = st.number_input(
            "연도",
            min_value=2000,
            max_value=2100,
            step=1,
            format="%d",
            value=default_year,
        )
        baseline_input = st.number_input(
            "온실가스 기준배출량(tCO2e)",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            value=float(default_baseline),
        )
        is_target_input = st.checkbox(
            "이행연도 대상 여부",
            value=default_is_target,
        )

        submitted = st.form_submit_button("저장")

    if submitted:
        year_int = int(year_input)
        records[year_int] = {
            "baseline": float(baseline_input),
            "is_target": bool(is_target_input),
        }
        baseline_mod.save_baseline_records(records, BASELINE_PATH)
        st.success(f"{year_int}년 기준배출량이 저장되었습니다.")
        # 저장 후 편집 상태 초기화 + 대시보드 재계산을 위해 rerun
        st.session_state["baseline_edit_year"] = None
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    # ----- 삭제 기능 -----
    st.markdown("#### 기준배출량 삭제")

    if records:
        delete_years = st.multiselect(
            "삭제할 연도 선택",
            options=sorted(records.keys(), reverse=True),
            format_func=lambda y: f"{y}년",
        )
        if st.button("선택 연도 삭제"):
            for y in delete_years:
                records.pop(y, None)
            baseline_mod.save_baseline_records(records, BASELINE_PATH)
            st.success("선택한 연도의 기준배출량을 삭제했습니다.")
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
    else:
        st.info("삭제할 기준배출량이 없습니다.")


# ============================================================
# 🔧 3) 디버그/진단 탭
# ============================================================

with tab_debug:
    st.header("데이터 구조 진단 (개발/테스트용)")

    st.caption(
        "data/energy 폴더에 저장된 모든 엑셀 파일에 대해 "
        "사전 구조 진단(validate_excel_file)을 수행합니다. "
        "새로운 양식을 적용하기 전에 이 탭에서 먼저 구조를 확인해 보세요."
    )

    energy_files = sorted(ENERGY_DIR.glob("*.xlsx"))
    if not energy_files:
        st.info("현재 data/energy 폴더에 .xlsx 파일이 없습니다.")
    else:
        st.write("#### 검사 대상 파일 목록")
        st.write(pd.DataFrame({"파일명": [p.name for p in energy_files]}))

    if st.button("data/energy 폴더 전체 구조 점검 실행"):
        results = []
        for xlsx_path in energy_files:
            v = loader.validate_excel_file(xlsx_path)
            issues_text = "\n".join(v["issues"]) if v["issues"] else ""
            warnings_text = "\n".join(v["warnings"]) if v["warnings"] else ""
            results.append(
                {
                    "파일명": v.get("filename", xlsx_path.name),
                    "OK": v["ok"],
                    "이슈_개수": len(v["issues"]),
                    "경고_개수": len(v["warnings"]),
                    "기관명_컬럼": v.get("detected_facility_col"),
                    "온실가스_컬럼": v.get("detected_ghg_col"),
                    "월별_컬럼_수": len(v.get("detected_month_cols", [])),
                    "이슈_요약": issues_text,
                    "경고_요약": warnings_text,
                }
            )

        if results:
            df_check = pd.DataFrame(results)
            st.write("#### 구조 진단 결과")
            st.dataframe(df_check)
        else:
            st.info("검사 결과가 없습니다.")
