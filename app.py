# app.py
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from pathlib import Path
import traceback  # 상세 오류 출력용

import pandas as pd
import streamlit as st

from modules import loader, analyzer, feedback


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


# ============================
# 헬퍼 함수들
# ============================

def ensure_dirs():
    """data/ 및 data/energy/ 폴더 보장."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    loader.ensure_energy_dir(ENERGY_DIR)


def load_baseline_map(path: Path = BASELINE_PATH):
    """
    baseline.json을 읽어 {연도: 기준배출량} 딕셔너리로 반환.
    파일이 없거나 형식이 안 맞으면 빈 dict 반환.
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    baseline_map = {}
    for year_str, info in raw.items():
        try:
            year = int(year_str)
            if isinstance(info, dict) and "baseline" in info:
                baseline_map[year] = float(info["baseline"])
        except Exception:
            continue

    return baseline_map


def load_all_energy_data(base_dir: Path = ENERGY_DIR):
    """
    data/energy/ 아래의 모든 .xlsx 파일을 읽어 표준 스키마 DF로 통합하고,
    파일 메타 정보와 에러 목록도 함께 반환.
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
    선택 연도에 대한 주요 지표를 한 번에 가져오기.
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
baseline_map = load_baseline_map()

# ============================
# 파일 업로드 UI + 저장/갱신
# ============================

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
            # 같은 파일명을 여러 번 처리하는 것 방지
            if f.name in st.session_state["processed_uploads"]:
                continue

            try:
                # loader 내부에서 엑셀 구조 사전 진단 + 정규화까지 수행
                _, year, saved_path = loader.process_uploaded_energy_file(
                    file_obj=f,
                    original_filename=f.name,
                    base_dir=ENERGY_DIR,
                )
                st.session_state["processed_uploads"].add(f.name)
                st.success(f"{f.name} (연도: {year}) 업로드 및 저장 완료")
                new_file_processed = True
            except loader.EnergyDataError as e:
                # 엑셀 구조 문제 등 사용자가 수정할 수 있는 에러
                st.error(f"{f.name} 업로드 처리 중 오류:\n{e}")
            except Exception as e:
                # 예기치 못한 오류
                st.error(f"{f.name} 업로드 처리 중 알 수 없는 오류가 발생했습니다: {e}")

# 새 파일을 처리했으면 앱을 다시 실행 (버전별 호환 처리)
if new_file_processed:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    # 둘 다 없으면 그냥 아래 코드 진행


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

# ============================
# 데이터 존재 여부 확인
# ============================

if df_all is None or df_all.empty:
    st.warning("아직 분석할 에너지 사용량 데이터가 없습니다. 상단에서 파일을 업로드해 주세요.")
    # 데이터 진단 섹션은 아래에서 계속 쓸 수 있도록 stop 하지 않고 바로 아래 코드로 가지 않도록 return
    st.stop()

# ============================
# analyzer 연동: 집계 데이터셋
# ============================

try:
    datasets = analyzer.build_dashboard_datasets(df_all, baseline_map=baseline_map)
except Exception as e:
    st.error(f"데이터 집계 중 오류가 발생했습니다: {e}")

    # 개발/디버그용 상세 스택 트레이스
    with st.expander("자세한 오류 정보 보기 (개발용)"):
        st.code(traceback.format_exc())

    st.stop()

monthly_by_agency = datasets["monthly_by_agency"]
monthly_total = datasets["monthly_total"]
annual_by_agency = datasets["annual_by_agency"]
annual_total = datasets["annual_total"]
annual_total_with_baseline = datasets["annual_total_with_baseline"]

# ============================
# 필터 UI (사이드바)
# ============================

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

# ============================
# 상단 KPI 카드
# ============================

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


# ============================
# 메인 레이아웃 (좌: 월별, 우: 5개년)
# ============================

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


# ============================
# 분석 · 피드백 영역
# ============================

st.markdown("---")
st.markdown("### 분석 · 피드백")

if actual_emission is None:
    st.info("선택한 연도에 대한 연간 배출량 정보가 없어, 분석·피드백을 생성할 수 없습니다.")
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
        baseline_emission=baseline_emission,
        reduction_rate_pct=reduction_rate_pct,
        ratio_to_baseline=ratio_to_baseline,
        recent_total_df=recent_total_df,
        current_month=current_month,
    )

    st.write(feedback_text)

# ============================
# 데이터 구조 진단 (개발/테스트용)
# ============================

st.markdown("---")
st.markdown("### 데이터 구조 진단 (개발/테스트용)")

st.caption(
    "data/energy 폴더에 저장된 모든 엑셀 파일에 대해 "
    "사전 구조 진단(validate_excel_structure)을 수행합니다. "
    "새로운 양식을 적용하기 전에 이 섹션에서 먼저 점검해 보세요."
)

if st.button("data/energy 폴더 전체 구조 점검 실행"):
    results = []
    for xlsx_path in sorted(ENERGY_DIR.glob("*.xlsx")):
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
        st.dataframe(df_check)
    else:
        st.info("data/energy 폴더에 검사할 엑셀 파일이 없습니다.")

# (옵션) 표준 스키마 미리보기
with st.expander("표준 스키마 데이터 미리보기 (디버깅용)"):
    st.write(df_all.head())
    st.caption("※ loader.normalize_energy_dataframe() 결과를 concat한 전체 데이터입니다.")
