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

    df_all = pd.concat(dfs, ignore_index=True) if dfs else None
    return df_all, meta_list, errors

# ======================================
# 원본 에너지 파일(U/V/W) 로딩 (신규)
# ======================================
def load_raw_year_data(year: int):
    """해당 연도의 원본 에너지사용량관리.xlsx를 원본 구조로 로드."""
    for p in ENERGY_DIR.glob("*.xlsx"):
        if str(year) in p.name:
            return loader.load_energy_raw_for_analysis(p)
    return None  # 파일 없음

# ============================
# 세션 상태 초기화
# ============================

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

ensure_dirs()

baseline_records = baseline_mod.load_baseline_records(BASELINE_PATH)
baseline_map = baseline_mod.get_baseline_map(baseline_records)

# ============================
# 화면 탭 구성
# ============================

tab_dashboard, tab_baseline, tab_debug = st.tabs(
    ["📊 대시보드", "⚙️ 기준배출량 관리", "🔧 디버그/진단"]
)

# ============================================================
# 📊 1) 대시보드 탭
# ============================================================

with tab_dashboard:

    # -----------------------------
    # 🔧 개발 진행 상황 표시 (요청사항)
    # -----------------------------
    with st.expander("🛠️ 현재 진행 중인 기능 반영 현황 (정보 표시용)"):
        st.markdown("""
        # 🔧 기능 반영 현황 요약

        ## 1. 기존 기능 변경
        - 기존 **전망분석 섹션 전체 삭제**
        - 기존 **피드백 섹션 삭제**, 단  
          → 마지막 "**공단 전체 분석·코멘트**"는 유지됨

        ## 2. 신규 기능 – 에너지 사용량 분석 (U/V/W 기반)
        - 공단 전체 에너지 사용량(U열 합계)
        - 면적당 온실가스 배출량(V열 합계)
        - 3개년 평균 대비 증감률 (비교 불가 처리)
        - 시설군별(W열 평균) 분석 (의료 / 복지 / 기타)
        - 소속기구별 에너지 사용 분석(분포 비율, 평균 대비 비율, 증가율 등)

        ## 3. 에너지 기반 피드백
        - 공단 전체: 현재 월 / 목표달성을 위한 감축률
        - 소속기구별: 사용 분포 순위 / 3개년 증가율 순위 / 평균 대비 사용비율 순위 /
          권장 감축량 / 증가 사유 제출 대상

        ## 4. 공통 기능
        - 기관 순서 고정 정렬 적용
        - 모든 표 전체폭 배치 (좌우 분할 제거)

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
            help="예: 2024년 에너지 사용량관리.xlsx",
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
                    st.success(f"{f.name} (연도 {year}) 업로드 완료")
                    new_file_processed = True

                except Exception as e:
                    st.error(f"{f.name} 업로드 처리 중 오류: {e}")

        if new_file_processed:
            st.rerun()

    with upload_col2:
        st.markdown("#### 저장된 연도별 파일 목록")

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
    # 집계 계산 (온실가스 KPI 전용)
    # -----------------------------
    datasets = analyzer.build_dashboard_datasets(df_all, baseline_map)

    # -----------------------------
    # 필터
    # -----------------------------
    years = sorted(df_all["연도"].unique().tolist())
    selected_year = max(years)

    st.sidebar.header("필터")
    st.sidebar.markdown("**기관 선택 제거됨 → 공단 전체 기준 고정**")

    selected_year = st.sidebar.selectbox("연도 선택", years, index=years.index(selected_year))

    # ============================
    # 🔥 신규 기능 1 — 주요지표
    # ============================

    st.markdown("## 에너지 사용량 추이")

    raw_df = load_raw_year_data(selected_year)
    if raw_df is None:
        st.error(f"{selected_year}년 원본 파일을 찾을 수 없습니다.")
        st.stop()

    # 원본 컬럼 참조
    org_col = raw_df.columns[2]   # C열
    U_col   = raw_df.columns[20]  # U열
    V_col   = raw_df.columns[21]  # V열
    W_col   = raw_df.columns[22]  # W열

    # 공단 전체 U/V/W 계산
    total_U = raw_df[U_col].sum(skipna=True)
    total_V = raw_df[V_col].sum(skipna=True)

    # 3개년 평균 U 계산
    past_years = [selected_year-3, selected_year-2, selected_year-1]
    past_vals = []
    for y in past_years:
        df_past = load_raw_year_data(y)
        if df_past is not None:
            past_vals.append(df_past[df_past.columns[20]].sum(skipna=True))

    if len(past_vals) >= 1:
        past_avg = sum(past_vals)/len(past_vals)
        U_change_rate = (total_U - past_avg) / past_avg * 100 if past_avg else None
    else:
        past_avg = None
        U_change_rate = None  # 비교불가

    k1, k2, k3 = st.columns(3)
    k1.metric("에너지 사용량(U합계)", f"{total_U:,.0f}")
    k2.metric("면적당 온실가스 배출량(V합계)", f"{total_V:,.0f}")
    k3.metric("3개년 평균 대비 증감률", "-" if U_change_rate is None else f"{U_change_rate:,.1f}%")

    # ============================
    # 신규 기능 2 — 시설군별 평균(W열)
    # ============================

    st.markdown("### 시설군별 평균 에너지 사용량(W열 기준)")

    MEDICAL = ["중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원"]
    WELFARE = ["수원요양원","광주요양원","김해요양원","대구요양원","대전요양원","남양주요양원","원주요양원","전주요양원"]
    OTHER   = ["본사","교육연구원","보훈원","재활체육센터","휴양원"]

    def avg_group(names):
        return raw_df[raw_df[org_col].isin(names)][W_col].mean()

    wg1, wg2, wg3 = st.columns(3)
    wg1.metric("의료시설 평균(W)", f"{avg_group(MEDICAL):,.1f}")
    wg2.metric("복지시설 평균(W)", f"{avg_group(WELFARE):,.1f}")
    wg3.metric("기타시설 평균(W)", f"{avg_group(OTHER):,.1f}")

    # ============================
    # 신규 기능 3 — 소속기구별 분석 표
    # ============================

    st.markdown("## 소속기구별 에너지 사용 분석")

    df_group = raw_df.groupby(org_col).agg(
        U합계=(U_col, "sum"),
        V합계=(V_col, "sum"),
        W평균=(W_col, "mean"),
    ).reset_index().rename(columns={org_col: "기관명"})

    # 시설구분
    def facility_type(name):
        if name in MEDICAL: return "의료시설"
        if name in WELFARE: return "복지시설"
        if name in OTHER:   return "기타시설"
        return "기타시설"

    df_group["시설구분"] = df_group["기관명"].apply(facility_type)

    # 공단 전체 대비 분포비율
    df_group["분포비율"] = df_group["U합계"] / total_U * 100 if total_U else None

    # 시설군별 평균 대비 비율
    med_avg = avg_group(MEDICAL)
    wel_avg = avg_group(WELFARE)
    oth_avg = avg_group(OTHER)

    def avg_compare(row):
        if row["시설구분"]=="의료시설":
            return row["W평균"]/med_avg if med_avg else None
        elif row["시설구분"]=="복지시설":
            return row["W평균"]/wel_avg if wel_avg else None
        else:
            return row["W평균"]/oth_avg if oth_avg else None

    df_group["평균대비사용비율"] = df_group.apply(avg_compare, axis=1)

    # 3개년 평균 대비 증감률
    def three_year_rate(name):
        past_vals=[]
        for y in past_years:
            dfp = load_raw_year_data(y)
            if dfp is not None:
                val = dfp[dfp[dfp.columns[2]]==name][dfp.columns[20]].sum()
                past_vals.append(val)
        if len(past_vals)>=1:
            avg_p = sum(past_vals)/len(past_vals)
            if avg_p==0: return None
            now = df_group[df_group["기관명"]==name]["U합계"].iloc[0]
            return (now-avg_p)/avg_p*100
        return None

    df_group["3개년증감률"] = df_group["기관명"].apply(three_year_rate)

    # 기관 출력 순서 적용
    ORDER = ["본사","중앙병원","부산병원","광주병원","대구병원","대전병원","인천병원",
             "교육연구원","보훈원","수원요양원","광주요양원","김해요양원","대구요양원",
             "대전요양원","남양주요양원","원주요양원","전주요양원","재활체육센터","휴양원"]

    df_group["기관명"] = pd.Categorical(df_group["기관명"], categories=ORDER, ordered=True)
    df_group = df_group.sort_values("기관명")

    st.dataframe(df_group, use_container_width=True)

    # ============================
    # 신규 기능 4 — 에너지 기반 피드백
    # ============================

    st.markdown("## 에너지 기반 피드백")

    # 현재 월 (표준 스키마 df_all 이용)
    df_sel = df_all[df_all["연도"]==selected_year]
    current_month = int(df_sel["월"].max()) if not df_sel.empty else None

    # 목표달성 감축률 (V합계 / 기준배출량)
    baseline = baseline_map.get(selected_year)
    reduction_ratio = total_V / baseline * 100 if baseline else None

    fb1, fb2 = st.columns(2)
    fb1.metric("현재 월", f"{current_month}월" if current_month else "-")
    fb2.metric("목표달성 감축률(V/기준)", "-" if reduction_ratio is None else f"{reduction_ratio:,.1f}%")

    # 기관별 피드백 확장 (표)
    st.markdown("### 기관별 에너지 피드백(순위·권장감축 등)")

    df_fb = df_group.copy()

    # 사용 분포 순위
    df_fb["사용분포순위"] = df_fb["U합계"].rank(method="dense", ascending=False)

    # 3개년 평균 증가율 순위
    df_fb["3개년증가순위"] = df_fb["3개년증감률"].rank(method="dense", ascending=False)

    # 평균대비 사용비율 순위
    df_fb["평균대비순위"] = df_fb["평균대비사용비율"].rank(method="dense", ascending=False)

    # 권장 감축량 = 전체 필요감축량*(기관U비중)
    if baseline:
        need = total_V - baseline
        need = need if need>0 else 0
        df_fb["권장감축량"] = need * (df_fb["U합계"]/total_U)
    else:
        df_fb["권장감축량"] = None

    # 증가사유 제출 조건 (증감률>0 또는 평균보다 높은 비율)
    def need_reason(row):
        if (row["3개년증감률"] is not None and row["3개년증감률"]>0) or \
           (row["평균대비사용비율"] is not None and row["평균대비사용비율"]>1):
            return "O"
        return "X"

    df_fb["증가사유제출"] = df_fb.apply(need_reason,axis=1)

    st.dataframe(df_fb, use_container_width=True)

    # ============================
    # 기존 유지 구간 – 공단 전체 분석 코멘트
    # ============================

    st.markdown("## 공단 전체 분석·코멘트 (기존 유지)")

    actual_emission = analyzer.get_annual_ghg(df_all,by_agency=False).query("연도==@selected_year")["연간 온실가스 배출량"].sum()
    recent_df,_ = analyzer.get_recent_years_ghg(analyzer.get_annual_ghg(df_all,by_agency=False), base_year=selected_year)

    fb_text = feedback.generate_overall_feedback(
        year=selected_year,
        actual_emission=actual_emission,
        baseline_emission=baseline,
        reduction_rate_pct=None,
        ratio_to_baseline=None,
        recent_total_df=recent_df,
        current_month=current_month,
    )
    st.write(fb_text)

# ============================================================
# 2) 기준배출량 관리 탭 (기존 유지)
# ============================================================

with tab_baseline:
    st.header("기준배출량 관리")
    # (기존 코드 그대로 유지)
