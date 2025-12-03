# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path

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

# 기관 순서 및 시설군 정의 (공통 사용)
FACILITY_ORDER = [
    "본사",
    "중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원",
    "교육연구원", "보훈원",
    "수원요양원", "광주요양원", "김해요양원", "대구요양원",
    "대전요양원", "남양주요양원", "원주요양원", "전주요양원",
    "재활체육센터", "휴양원",
]

MEDICAL_FACILITIES = ["중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원"]
WELFARE_FACILITIES = ["수원요양원", "광주요양원", "김해요양원", "대구요양원",
                      "대전요양원", "남양주요양원", "원주요양원", "전주요양원"]
OTHER_FACILITIES = ["본사", "교육연구원", "보훈원", "재활체육센터", "휴양원"]


# ============================
# 데이터 로딩 헬퍼
# ============================

def load_all_energy_data(base_dir: Path = ENERGY_DIR):
    """
    저장된 모든 연도 파일을 로드하여
    - 표준 스키마 데이터 df_all
    - 파일 메타 정보
    - 로딩 오류 목록
    을 반환한다.

    주의: ensure_dirs() 호출 금지
    """
    dfs = []
    meta_list = []
    errors = []

    for xlsx_path in sorted(base_dir.glob("*.xlsx")):
        try:
            df_std, year = loader.load_energy_xlsx(xlsx_path)
            dfs.append(df_std)

            stat = xlsx_path.stat()
            meta_list.append({
                "연도": year,
                "파일명": xlsx_path.name,
                "경로": str(xlsx_path),
                "업로드시간": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })
        except loader.EnergyDataError as e:
            errors.append({"파일명": xlsx_path.name, "에러": str(e)})
        except Exception as e:
            errors.append({"파일명": xlsx_path.name, "에러": f"알 수 없는 오류: {e}"})

    df_all = pd.concat(dfs, ignore_index=True) if dfs else None
    return df_all, meta_list, errors


# ======================================
# U/V/W 원본 분석용 로딩 함수
# ======================================

def load_raw_year_data(year: int):
    for p in ENERGY_DIR.glob("*.xlsx"):
        if str(year) in p.name:
            return loader.load_energy_raw_for_analysis(p)
    return None


# ============================
# 세션 상태
# ============================

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

# baseline 로드
baseline_records = baseline_mod.load_baseline_records(BASELINE_PATH)
baseline_map = baseline_mod.get_baseline_map(baseline_records)


# ============================
# 탭 구성
# ============================

tab_dashboard, tab_baseline, tab_debug = st.tabs(
    ["📊 대시보드", "⚙️ 기준배출량 관리", "🔧 디버그/진단"]
)

# ============================================================
# 📊 1) 대시보드 탭
# ============================================================

with tab_dashboard:

    # -----------------------------
    # 진행중 기능 반영 현황 표시
    # -----------------------------
    with st.expander("🛠️ 현재 진행 중인 기능 반영 현황"):
        st.markdown("""
        # 🔧 기능 반영 현황

        **1. 기존 기능 변경**
        - 기존 전망분석 섹션 삭제
        - 기존 피드백 섹션 대부분 제거 (마지막 전체 코멘트만 유지)

        **2. 신규 기능 — 에너지 사용량 분석(U/V/W)**
        - 공단 전체 에너지 사용량(U)
        - 면적당 온실가스 배출량(V)
        - 3개년 평균 대비 증감률
        - 시설군별(W열 평균) 분석
        - 소속기구별 에너지 분석 표

        **3. 신규 — 에너지 기반 피드백**
        - 공단 전체: 현재 월 / 목표달성 감축률(V 대비 기준배출량)
        - 소속기구별: 분포순위 / 증가율 / 평균 대비 / 권장 감축량 / 증가 사유 제출

        **4. 공통 기능**
        - 기관 순서 고정
        - 표 전체폭으로 출력
        """)

    # ------------------------------
    # 파일 업로드
    # ------------------------------
    st.markdown("### 월별 에너지 사용량 파일 업로드")

    upload_col1, upload_col2 = st.columns([1.2, 2])
    new_file_processed = False

    with upload_col1:
        uploaded_files = st.file_uploader(
            "에너지 사용량관리 .xlsx 파일 업로드",
            type=["xlsx"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for f in uploaded_files:
                if f.name in st.session_state["processed_uploads"]:
                    continue
                try:
                    # 업로드 및 표준 스키마 변환
                    _, year, saved_path = loader.process_uploaded_energy_file(
                        file_obj=f,
                        original_filename=f.name,
                        base_dir=ENERGY_DIR,
                    )
                    st.session_state["processed_uploads"].add(f.name)
                    st.success(f"{f.name} ({year}) 업로드 완료")
                    new_file_processed = True
                except Exception as e:
                    st.error(f"업로드 오류: {e}")

        if new_file_processed:
            st.rerun()

    # 저장된 파일 목록
    with upload_col2:
        st.markdown("#### 저장된 파일 목록")

        df_all, files_meta, load_errors = load_all_energy_data()

        if files_meta:
            df_files = pd.DataFrame(files_meta).sort_values(
                ["연도", "업로드시간"], ascending=[False, False]
            )
            st.table(df_files[["연도", "파일명", "업로드시간"]])
        else:
            st.info("저장된 파일 없음")

    st.markdown("---")

    if df_all is None:
        st.warning("에너지 사용량 데이터가 없습니다.")
        st.stop()

    # -----------------------------
    # analyzer 기반 집계 데이터 패키지
    # (기존 '주요 지표 + 그래프'용)
    # -----------------------------
    datasets = analyzer.build_dashboard_datasets(df_all, baseline_map)
    annual_total = datasets["annual_total"]
    annual_total_with_baseline = datasets["annual_total_with_baseline"]
    monthly_total = datasets["monthly_total"]
    monthly_by_agency = datasets["monthly_by_agency"]
    annual_by_agency = datasets["annual_by_agency"]

    years = sorted(df_all["연도"].dropna().unique().tolist())
    default_year = max(years)

    # ============================================================
    # 1) 에너지 사용량 추이 (기존 "주요 지표" 영역)
    # ============================================================

    st.markdown("## 에너지 사용량 추이")

    # 좌측 필터 + 우측 지표/그래프 레이아웃
    filter_col, main_col = st.columns([1, 3])

    # -------- 좌측 필터 영역 (기존 유지) --------
    with filter_col:
        st.subheader("필터")

        view_scope = st.radio("보기 범위", ["공단 전체", "기관별"], index=0)

        # 이행연도 선택 = 전체 페이지 공통 기준 연도
        selected_year = st.selectbox(
            "이행연도 선택",
            years,
            index=years.index(default_year),
        )

        # 기관 선택 (기관별 선택 시)
        selected_org = None
        if view_scope == "기관별":
            org_list = df_all["기관명"].dropna().unique().tolist()
            ordered = [o for o in FACILITY_ORDER if o in org_list]
            others = sorted([o for o in org_list if o not in FACILITY_ORDER])
            org_options = ordered + others
            if not org_options:
                st.warning("기관 정보가 없습니다.")
            else:
                selected_org = st.selectbox("기관 선택", org_options)

        st.markdown("에너지 종류 필터 (추후 확장용)")
        _ = st.selectbox("에너지 종류", ["전체"])

    # -------- 우측 주요 지표 + 그래프 --------
    # 선택 연도 기준 KPI 계산 (공단 기준)
    kpi_row = annual_total_with_baseline[
        annual_total_with_baseline["연도"] == selected_year
    ]

    if not kpi_row.empty:
        row0 = kpi_row.iloc[0]
        kpi_baseline = row0["기준배출량"]
        kpi_emission = row0["연간 온실가스 배출량"]
        kpi_ratio_pct = (
            row0["배출비율"] * 100 if pd.notna(row0["배출비율"]) else None
        )
        kpi_reduction_pct = row0["감축률(%)"]
    else:
        kpi_baseline = None
        kpi_emission = None
        kpi_ratio_pct = None
        kpi_reduction_pct = None

    # 그래프용 데이터 준비
    if view_scope == "공단 전체":
        monthly_df = monthly_total[monthly_total["연도"] == selected_year]
        recent_df, _ = analyzer.get_recent_years_ghg(
            annual_total, base_year=int(selected_year)
        )
    else:
        # 기관별
        if selected_org is not None:
            monthly_df = monthly_by_agency[
                (monthly_by_agency["연도"] == selected_year)
                & (monthly_by_agency["기관명"] == selected_org)
            ]
            annual_sel = annual_by_agency[
                annual_by_agency["기관명"] == selected_org
            ]
            recent_df, _ = analyzer.get_recent_years_ghg(
                annual_sel, base_year=int(selected_year)
            )
        else:
            monthly_df = pd.DataFrame()
            recent_df = pd.DataFrame()

    with main_col:
        # ----- 상단 KPI (기존 '주요 지표' 유지 + 기준배출량 표시) -----
        k1, k2, k3, k4 = st.columns(4)

        # 선택 연도 + 기준배출량
        if kpi_baseline is not None:
            k1.metric("선택 연도", f"{selected_year}년")
            k1.caption(f"기준배출량: {kpi_baseline:,.0f} tCO2eq")
        else:
            k1.metric("선택 연도", f"{selected_year}년")
            k1.caption("기준배출량 미등록")

        # 연간 온실가스 배출량(공단)
        if kpi_emission is not None:
            k2.metric("연간 온실가스 배출량(공단)", f"{kpi_emission:,.0f} tCO2eq")
        else:
            k2.metric("연간 온실가스 배출량(공단)", "-")

        # 감축률(전체 기준)
        if kpi_reduction_pct is not None:
            k3.metric("감축률(전체 기준)", f"{kpi_reduction_pct:,.1f} %")
        else:
            k3.metric("감축률(전체 기준)", "-")

        # 기준배출량 대비 배출비율
        if kpi_ratio_pct is not None:
            k4.metric("기준배출량 대비 배출비율", f"{kpi_ratio_pct:,.1f} %")
        else:
            k4.metric("기준배출량 대비 배출비율", "-")

        # ----- 그래프 (기존 구조 유지) -----
        # 이행연도 월별 온실가스 추이
        st.markdown("#### 이행연도 월별 온실가스 추이")
        if not monthly_df.empty:
            chart_month = (
                monthly_df.sort_values("월")[["월", "월별 온실가스 환산량"]]
                .set_index("월")
            )
            st.line_chart(chart_month)
        else:
            st.info("선택 조건에 해당하는 월별 데이터가 없습니다.")

        # 최근 5개년 연간 배출량 추이
        st.markdown("#### 최근 5개년 연간 배출량 추이")
        if not recent_df.empty:
            chart_recent = (
                recent_df.sort_values("연도")[["연도", "연간 온실가스 배출량"]]
                .set_index("연도")
            )
            st.bar_chart(chart_recent)
        else:
            st.info("선택 조건에 해당하는 연간 데이터가 없습니다.")

    # ============================================================
    # 2) 에너지 사용량 분석 (U/V/W 기반 신규 섹션)
    # ============================================================

    st.markdown("---")
    st.markdown("## 에너지 사용량 분석 (에너지 사용량관리 파일 기준)")

    raw_df = load_raw_year_data(int(selected_year))
    if raw_df is None:
        st.error(f"{selected_year}년 원본 파일을 찾을 수 없습니다.")
        st.stop()

    # 원본 컬럼 인덱스 (C, U, V, W)
    org_col = raw_df.columns[2]    # C열: 소속기구명
    U_col   = raw_df.columns[20]   # U열: 에너지 사용량
    V_col   = raw_df.columns[21]   # V열: 면적당 온실가스 배출량
    W_col   = raw_df.columns[22]   # W열: 평균 에너지 사용량(연면적 기준)

    # 기관명 / 수치 전처리: 공백 제거, NA 제거, 숫자 변환
    raw_df = raw_df[raw_df[org_col].notna()].copy()
    raw_df[org_col] = raw_df[org_col].astype(str).str.strip()

    for c in [U_col, V_col, W_col]:
        raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

    # 공단 전체기준 KPI
    total_U = raw_df[U_col].sum(skipna=True)
    total_V = raw_df[V_col].sum(skipna=True)

    # 3개년 평균 대비 증감률 (U열 기준)
    past_years = [int(selected_year) - 3, int(selected_year) - 2, int(selected_year) - 1]
    past_vals = []
    for y in past_years:
        df_past = load_raw_year_data(y)
        if df_past is not None:
            org_c = df_past.columns[2]
            U_c   = df_past.columns[20]
            df_past = df_past[df_past[org_c].notna()].copy()
            df_past[org_c] = df_past[org_c].astype(str).str.strip()
            df_past[U_c] = pd.to_numeric(df_past[U_c], errors="coerce")
            past_vals.append(df_past[U_c].sum(skipna=True))

    if past_vals:
        past_avg = sum(past_vals) / len(past_vals)
        U_change_rate = (total_U - past_avg) / past_avg * 100 if past_avg else None
    else:
        past_avg = None
        U_change_rate = None

    st.markdown("### 공단 전체 기준")

    k1, k2, k3 = st.columns(3)
    k1.metric("에너지 사용량(U 합계)", f"{total_U:,.0f}")
    k2.metric("면적당 온실가스 배출량(V 합계)", f"{total_V:,.0f}")
    k3.metric("3개년 평균 대비 증감률", "-" if U_change_rate is None else f"{U_change_rate:,.1f}%")

    # 시설군별 평균 에너지 사용량(W열)
    st.markdown("### 시설군별 평균 에너지 사용량(W열)")

    def avg_group(names):
        df_tmp = raw_df[raw_df[org_col].isin(names)]
        return df_tmp[W_col].mean(skipna=True)

    g1, g2, g3 = st.columns(3)
    g1.metric("의료시설 평균(W)", f"{avg_group(MEDICAL_FACILITIES):,.1f}")
    g2.metric("복지시설 평균(W)", f"{avg_group(WELFARE_FACILITIES):,.1f}")
    g3.metric("기타시설 평균(W)", f"{avg_group(OTHER_FACILITIES):,.1f}")

    # 소속기구별 에너지 사용 분석
    st.markdown("### 소속기구별")

    df_group = raw_df.groupby(org_col).agg(
        U합계=(U_col, "sum"),
        V합계=(V_col, "sum"),
        W평균=(W_col, "mean"),
    ).reset_index().rename(columns={org_col: "기관명"})

    # 시설구분 부여
    def facility_type(name: str) -> str:
        if name in MEDICAL_FACILITIES:
            return "의료시설"
        if name in WELFARE_FACILITIES:
            return "복지시설"
        if name in OTHER_FACILITIES:
            return "기타시설"
        return "기타시설"

    df_group["시설구분"] = df_group["기관명"].apply(facility_type)

    # 공단 전체 대비 분포비율
    df_group["분포비율"] = df_group["U합계"] / total_U * 100 if total_U else None

    # 시설군별 평균 대비 비율
    med_avg = avg_group(MEDICAL_FACILITIES)
    wel_avg = avg_group(WELFARE_FACILITIES)
    oth_avg = avg_group(OTHER_FACILITIES)

    def avg_compare(row):
        if row["시설구분"] == "의료시설":
            return row["W평균"] / med_avg if med_avg else None
        if row["시설구분"] == "복지시설":
            return row["W평균"] / wel_avg if wel_avg else None
        return row["W평균"] / oth_avg if oth_avg else None

    df_group["평균대비사용비율"] = df_group.apply(avg_compare, axis=1)

    # 3개년 평균 에너지 사용 증감률 (기관별)
    def three_year_rate(name: str):
        past_vals = []
        for y in past_years:
            dfp = load_raw_year_data(y)
            if dfp is not None:
                org_c = dfp.columns[2]
                U_c = dfp.columns[20]
                dfp = dfp[dfp[org_c].notna()].copy()
                dfp[org_c] = dfp[org_c].astype(str).str.strip()
                dfp[U_c] = pd.to_numeric(dfp[U_c], errors="coerce")
                val = dfp[dfp[org_c] == name][U_c].sum(skipna=True)
                past_vals.append(val)

        if past_vals:
            avg_p = sum(past_vals) / len(past_vals)
            now = df_group[df_group["기관명"] == name]["U합계"].iloc[0]
            if avg_p > 0:
                return (now - avg_p) / avg_p * 100
        return None

    df_group["3개년증감률"] = df_group["기관명"].apply(three_year_rate)

    # 기관 순서 고정
    df_group["기관명"] = pd.Categorical(
        df_group["기관명"], categories=FACILITY_ORDER, ordered=True
    )
    df_group = df_group.sort_values("기관명")

    st.dataframe(df_group, use_container_width=True)

    # ============================================================
    # 3) 에너지 기반 피드백 (신규)
    # ============================================================

    st.markdown("## 에너지 기반 피드백")

    # 현재 월 (표준 스키마 기준)
    df_sel = df_all[df_all["연도"] == selected_year]
    current_month = int(df_sel["월"].max()) if not df_sel.empty else None

    # 목표 달성을 위한 감축률 분석: V합계 / 기준배출량
    baseline_val = baseline_map.get(int(selected_year))
    reduction_ratio = total_V / baseline_val * 100 if baseline_val else None

    st.markdown("### 공단 전체 기준")
    f1, f2 = st.columns(2)
    f1.metric("현재 월", f"{current_month}월" if current_month else "-")
    f2.metric("목표달성 감축률(V/기준)", "-" if reduction_ratio is None else f"{reduction_ratio:,.1f}%")

    st.markdown("### 기관별 피드백")

    df_fb = df_group.copy()

    # 사용 분포 순위 (U합계 기준)
    df_fb["사용분포순위"] = df_fb["U합계"].rank(ascending=False, method="dense")

    # 3개년 평균 증가 순위
    df_fb["3개년증가순위"] = df_fb["3개년증감률"].rank(ascending=False, method="dense")

    # 평균 에너지 사용량(연면적 기준) 순위
    df_fb["평균대비순위"] = df_fb["평균대비사용비율"].rank(ascending=False, method="dense")

    # 목표 권장 감축량 (공단 전체 추가 감축 필요량을 기관별 비중으로 배분)
    if baseline_val:
        need_total = total_V - baseline_val
        need_total = need_total if need_total > 0 else 0
        df_fb["권장감축량"] = need_total * (df_fb["U합계"] / total_U) if total_U else 0
    else:
        df_fb["권장감축량"] = None

    # 에너지 사용량 증가 사유 제출 대상
    def need_reason(row):
        cond1 = (row["3개년증감률"] is not None) and (row["3개년증감률"] > 0)
        cond2 = (row["평균대비사용비율"] is not None) and (row["평균대비사용비율"] > 1)
        return "O" if (cond1 or cond2) else "X"

    df_fb["증가사유제출"] = df_fb.apply(need_reason, axis=1)

    st.dataframe(df_fb, use_container_width=True)

    # ============================================================
    # 4) 기존 유지 — 공단 전체 분석·코멘트
    # ============================================================

    st.markdown("## 공단 전체 분석·코멘트")

    annual_total_only = analyzer.get_annual_ghg(df_all, by_agency=False)
    actual_emission = annual_total_only.query("연도 == @selected_year")["연간 온실가스 배출량"].sum()

    recent_total_df, _ = analyzer.get_recent_years_ghg(
        annual_total_only,
        base_year=int(selected_year),
    )

    fb_text = feedback.generate_overall_feedback(
        year=int(selected_year),
        actual_emission=actual_emission,
        baseline_emission=baseline_val,
        reduction_rate_pct=None,
        ratio_to_baseline=None,
        recent_total_df=recent_total_df,
        current_month=current_month,
    )

    st.write(fb_text)


# ============================================================
# ⚙️ 2) 기준배출량 관리 탭
# ============================================================

with tab_baseline:
    st.header("기준배출량 관리")

    st.markdown("### 현재 기준배출량 목록")
    df_b = pd.DataFrame(baseline_records)
    if not df_b.empty:
        st.table(df_b)
    else:
        st.info("등록된 기준배출량 없음")

    st.markdown("### 기준배출량 신규 등록")
    col1, col2 = st.columns(2)

    new_year = col1.number_input("연도", min_value=2000, max_value=2100, step=1)
    new_val = col2.number_input("기준배출량(tCO2eq)", min_value=0.0, step=100.0)

    if st.button("저장"):
        baseline_mod.update_baseline_record(BASELINE_PATH, new_year, new_val)
        st.success("기준배출량 저장 완료")
        st.rerun()


# ============================================================
# 🔧 3) 디버그 / 진단 탭
# ============================================================

with tab_debug:

    st.header("디버그 / 구조 진단")

    st.markdown("### 파일 구조 진단")
    uploaded_debug_file = st.file_uploader("엑셀 구조 진단 파일 업로드 (.xlsx)", type=["xlsx"])
    if uploaded_debug_file:
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_debug_file.read())
            tmp_path = Path(tmp.name)

        try:
            res = loader.validate_excel_file(tmp_path)
            st.json(res)
        except Exception as e:
            st.error(f"진단 오류: {e}")

    st.markdown("---")

    # 실행 환경 진단 — loader.py 확인
    with st.expander("🧪 실행 환경 진단: loader.py 확인"):
        import modules.loader as ld
        import inspect

        st.subheader("📌 Streamlit이 사용 중인 loader.py 경로")
        st.code(ld.__file__)

        st.subheader("📌 함수 목록")
        st.write(dir(ld))

        st.subheader("📌 실제 loader.py 소스 코드")
        try:
            st.code(inspect.getsource(ld), language="python")
        except Exception:
            st.error("소스 코드를 불러올 수 없습니다.")
