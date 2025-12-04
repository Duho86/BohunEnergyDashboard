# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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


# 기관 순서 및 시설군 정의 (대시보드 공통)
FACILITY_ORDER = [
    "본사",
    "중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원",
    "교육연구원", "보훈원",
    "수원요양원", "광주요양원", "김해요양원", "대구요양원",
    "대전요양원", "남양주요양원", "원주요양원", "전주요양원",
    "재활체육센터", "휴양원",
]

MEDICAL_FACILITIES = [
    "중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원"
]
WELFARE_FACILITIES = [
    "수원요양원", "광주요양원", "김해요양원", "대구요양원",
    "대전요양원", "남양주요양원", "원주요양원", "전주요양원",
]
OTHER_FACILITIES = [
    "본사", "교육연구원", "보훈원", "재활체육센터", "휴양원"
]


# ============================
# 공통 유틸
# ============================

def load_all_energy_data(base_dir: Path = ENERGY_DIR):
    """
    저장된 모든 연도 파일을 로드하여
    - 표준 스키마 데이터 df_all
    - 파일 메타 정보
    - 로딩 오류 목록
    을 반환한다.
    """
    dfs: List[pd.DataFrame] = []
    meta_list: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

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


def load_raw_year_data(year: int) -> pd.DataFrame | None:
    """
    '에너지 사용량관리.xlsx' 원본 구조를 그대로 읽어오는 함수.
    (U/V/W 및 월별 데이터 분석용)
    """
    for p in ENERGY_DIR.glob("*.xlsx"):
        if str(year) in p.name:
            return loader.load_energy_raw_for_analysis(p)
    return None


def preprocess_uv_w(df_raw: pd.DataFrame):
    """
    U/V/W 열 및 기관명에 대해
    - 기관명 공백 제거
    - U/V/W → float 변환
    - 변환 실패 값은 오류 리스트에 기록 후 NaN 처리
    (집계 시 NaN은 자동 제외)
    """
    errors: List[Dict[str, Any]] = []

    org_col = df_raw.columns[2]
    U_col = df_raw.columns[20]
    V_col = df_raw.columns[21]
    W_col = df_raw.columns[22]

    df = df_raw.copy()

    # 기관명 정제
    df = df[df[org_col].notna()].copy()
    df[org_col] = df[org_col].astype(str).str.strip()

    def _to_numeric_with_log(series: pd.Series, col_label: str) -> pd.Series:
        s_raw = series
        s_str = s_raw.astype(str).str.strip()

        # 완전 공백/빈문자열은 결측으로 처리
        empty_mask = s_str == ""
        s_str = s_str.mask(empty_mask, pd.NA)

        converted = pd.to_numeric(s_str, errors="coerce")

        # 변환 오류(숫자로 해석 불가) 로깅
        err_mask = s_str.notna() & converted.isna()
        if err_mask.any():
            for idx in s_raw[err_mask].index:
                errors.append({
                    "row": int(idx),
                    "컬럼": str(col_label),
                    "값": s_raw.loc[idx],
                })
        return converted

    df[U_col] = _to_numeric_with_log(df[U_col], U_col)
    df[V_col] = _to_numeric_with_log(df[V_col], V_col)
    df[W_col] = _to_numeric_with_log(df[W_col], W_col)

    return df, org_col, U_col, V_col, W_col, errors


def detect_last_month_with_data(df_raw: pd.DataFrame) -> int | None:
    """
    '에너지 사용량 관리' 원본에서
    - '1월' ~ '12월' 컬럼 중
    - 실제 숫자 데이터가 존재하는 가장 마지막 월 번호를 반환.
    """
    last_month: int | None = None

    month_cols = [
        c for c in df_raw.columns
        if isinstance(c, str) and c.endswith("월") and c[0].isdigit()
    ]

    for c in month_cols:
        s_raw = df_raw[c]
        s_str = s_raw.astype(str).str.strip()
        empty_mask = s_str == ""
        s_str = s_str.mask(empty_mask, pd.NA)
        converted = pd.to_numeric(s_str, errors="coerce")

        if converted.notna().any():
            try:
                m = int(str(c).replace("월", ""))
                if (last_month is None) or (m > last_month):
                    last_month = m
            except ValueError:
                continue

    return last_month


# ============================
# 세션 상태
# ============================

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

# baseline 로드 (사용자 입력값만 사용)
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

        **1. 기존 기능 유지**
        - 상단 에너지 사용량 추이 영역(연도 선택, 기준배출량, 그래프 2개) 레이아웃 유지

        **2. 에너지 사용량 분석(신규)**
        - 공단 전체 기준(U/V/W 기반)
        - 소속기구별 에너지 사용량 및 분포/증감률 분석

        **3. 에너지 기반 피드백(신규)**
        - 공단 전체: 기준 달 / 목표달성을 위한 감축률 분석
        - 소속기구별: 사용 분포 순위 / 3개년 평균 증가 순위 /
          평균 에너지 사용량(연면적 기준) 순위 / 목표 권장 감축량 / 증가 사유 제출 여부

        **4. 공통**
        - 기관 순서 고정
        - 표는 화면 전체 폭으로 출력
        - None / NaN은 '-'로 표시
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
    #  (상단 에너지 사용량 추이 영역 용)
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
    # 1) 에너지 사용량 추이 (기존 상단 영역 유지)
    # ============================================================

    st.markdown("## 에너지 사용량 추이")

    filter_col, main_col = st.columns([1, 3])

    # ----- 좌측 필터 -----
    with filter_col:
        st.subheader("필터")

        view_scope = st.radio("보기 범위", ["공단 전체", "기관별"], index=0)

        selected_year = st.selectbox(
            "이행연도 선택",
            years,
            index=years.index(default_year),
        )

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

    # ----- 우측 주요지표 + 그래프 -----
    with main_col:
        # 선택 연도 기준 KPI (공단 전체 기준)
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

        k1, k2, k3, k4 = st.columns(4)

        # 선택 연도 + 기준배출량
        if kpi_baseline is not None:
            k1.metric("선택 연도", f"{selected_year}년")
            k1.caption(f"기준배출량: {kpi_baseline:,.0f} tCO2eq")
        else:
            k1.metric("선택 연도", f"{selected_year}년")
            k1.caption("기준배출량 미등록")

        # 연간 온실가스 배출량
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

        # 그래프용 데이터
        if view_scope == "공단 전체":
            monthly_df = monthly_total[monthly_total["연도"] == selected_year]
            recent_df, _ = analyzer.get_recent_years_ghg(
                annual_total, base_year=int(selected_year)
            )
        else:
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

        # 그래프 2개 좌우 배치
        st.markdown("")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 이행연도 월별 온실가스 추이")
            if not monthly_df.empty:
                chart_month = (
                    monthly_df.sort_values("월")[["월", "월별 온실가스 환산량"]]
                    .set_index("월")
                )
                st.line_chart(chart_month)
            else:
                st.info("선택 조건에 해당하는 월별 데이터가 없습니다.")

        with c2:
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
    # 2) 에너지 사용량 분석 (에너지 사용량 관리 엑셀 기준)
    # ============================================================

    st.markdown("---")
    st.markdown("## 에너지 사용량 분석")

    raw_df_original = load_raw_year_data(int(selected_year))
    if raw_df_original is None:
        st.error(f"{selected_year}년 원본 파일을 찾을 수 없습니다.")
        st.stop()

    # U/V/W & 기관명 전처리 (데이터 정제 + 오류 로깅)
    raw_df, org_col, U_col, V_col, W_col, preprocess_errors = preprocess_uv_w(
        raw_df_original
    )

    # --- 3-1) 공단 전체 기준 ---

    # 공단 전체 에너지 사용량 / 면적당 배출량
    total_U = raw_df[U_col].sum(skipna=True)
    total_V = raw_df[V_col].sum(skipna=True)

    # 3개년 평균 대비 증감률 (U열 기준)
    past_years = [
        int(selected_year) - 3,
        int(selected_year) - 2,
        int(selected_year) - 1,
    ]
    past_u_values: List[float] = []

    for y in past_years:
        df_past_raw = load_raw_year_data(y)
        if df_past_raw is not None:
            df_past, p_org, p_U, p_V, p_W, err = preprocess_uv_w(df_past_raw)
            val = df_past[p_U].sum(skipna=True)
            past_u_values.append(val)

    if past_u_values:
        past_avg_U = sum(past_u_values) / len(past_u_values)
        U_change_rate = (
            (total_U - past_avg_U) / past_avg_U * 100 if past_avg_U else None
        )
    else:
        past_avg_U = None
        U_change_rate = None

    st.markdown("### 공단 전체 기준")

    k1, k2, k3 = st.columns(3)
    k1.metric("에너지 사용량(현재 기준)", f"{total_U:,.0f}")
    k2.metric("면적당 온실가스 배출량", f"{total_V:,.0f}")
    k3.metric(
        "3개년 평균 에너지 사용량 대비 증감률",
        "-" if U_change_rate is None else f"{U_change_rate:,.1f} %",
    )

    # 평균 에너지 사용량(연면적 기준, W열 기준)
    st.markdown("#### 평균 에너지 사용량(연면적 기준)")

    def avg_group(names: List[str]) -> float | None:
        df_tmp = raw_df[raw_df[org_col].isin(names)]
        if df_tmp.empty:
            return None
        return float(df_tmp[W_col].mean(skipna=True))

    g1, g2, g3 = st.columns(3)
    med_avg = avg_group(MEDICAL_FACILITIES)
    wel_avg = avg_group(WELFARE_FACILITIES)
    oth_avg = avg_group(OTHER_FACILITIES)

    g1.metric(
        "의료시설",
        "-" if med_avg is None else f"{med_avg:,.1f}",
    )
    g2.metric(
        "복지시설",
        "-" if wel_avg is None else f"{wel_avg:,.1f}",
    )
    g3.metric(
        "기타시설",
        "-" if oth_avg is None else f"{oth_avg:,.1f}",
    )

    # --- 3-2) 소속기구별 에너지 사용량 분석 표 ---

    st.markdown("### 소속기구별 에너지 사용량 분석 표")

    df_group = (
        raw_df.groupby(org_col)
        .agg(
            에너지사용량=(U_col, "sum"),
            면적당배출량=(V_col, "sum"),
            W평균=(W_col, "mean"),
        )
        .reset_index()
    )

    # 컬럼 이름/내용을 사양에 맞게 구성
    df_group = df_group.rename(columns={
        org_col: "구분",
        "에너지사용량": "에너지 사용량(현재 기준)",
        "면적당배출량": "면적당 온실가스 배출량",
    })

    # 시설구분
    def facility_type(name: str) -> str:
        if name in MEDICAL_FACILITIES:
            return "의료시설"
        if name in WELFARE_FACILITIES:
            return "복지시설"
        if name in OTHER_FACILITIES:
            return "기타시설"
        return "기타시설"

    df_group["시설구분"] = df_group["구분"].apply(facility_type)

    # 공단 에너지 사용량 분포 비율
    df_group["공단 에너지 사용량 분포 비율"] = (
        df_group["에너지 사용량(현재 기준)"] / total_U * 100 if total_U else pd.NA
    )

    # 시설군별 평균 대비 사용비율
    def avg_compare(row):
        if row["시설구분"] == "의료시설":
            return (
                row["W평균"] / med_avg if (med_avg is not None and med_avg != 0) else pd.NA
            )
        if row["시설구분"] == "복지시설":
            return (
                row["W평균"] / wel_avg if (wel_avg is not None and wel_avg != 0) else pd.NA
            )
        return (
            row["W평균"] / oth_avg if (oth_avg is not None and oth_avg != 0) else pd.NA
        )

    df_group["평균 에너지 사용량(연면적 기준) 대비 사용비율"] = df_group.apply(
        avg_compare, axis=1
    )

    # 3개년 평균 에너지 사용 증감률 (기관별)
    def three_year_rate(name: str):
        vals: List[float] = []
        for y in past_years:
            dfp_raw = load_raw_year_data(y)
            if dfp_raw is not None:
                dfp, p_org, p_U, p_V, p_W, err = preprocess_uv_w(dfp_raw)
                dfp = dfp[dfp[p_org].notna()].copy()
                dfp[p_org] = dfp[p_org].astype(str).str.strip()
                now_val = dfp[dfp[p_org] == name][p_U].sum(skipna=True)
                vals.append(float(now_val))

        if vals:
            avg_past = sum(vals) / len(vals)
            now_u = df_group[df_group["구분"] == name]["에너지 사용량(현재 기준)"]
            if not now_u.empty and avg_past > 0:
                return (now_u.iloc[0] - avg_past) / avg_past * 100
        return pd.NA

    df_group["3개년 평균 에너지 사용 증감률"] = df_group["구분"].apply(three_year_rate)

    # 기관 순서 고정
    df_group["구분"] = pd.Categorical(
        df_group["구분"], categories=FACILITY_ORDER, ordered=True
    )
    df_group = df_group.sort_values("구분")

    # NaN은 '-'로 표시하면서 전체폭으로 표시
    st.dataframe(
        df_group.style.format(na_rep="-"),
        use_container_width=True,
    )

    # ============================================================
    # 3) 에너지 기반 피드백
    # ============================================================

    st.markdown("## 피드백")

    # --- 4-1) 공단 전체 기준 ---

    st.markdown("### 공단 전체 기준")

    # 기준 달: 원본 엑셀에서 실제 값이 있는 마지막 월
    기준달 = detect_last_month_with_data(raw_df_original)

    baseline_val = baseline_map.get(int(selected_year))
    reduction_ratio = (
        total_V / baseline_val * 100 if (baseline_val and baseline_val != 0) else None
    )

    f1, f2 = st.columns(2)
    f1.metric("기준 달", f"{기준달}월" if 기준달 is not None else "-")
    f2.metric(
        "목표달성을 위한 감축률 분석",
        "-" if reduction_ratio is None else f"{reduction_ratio:,.1f} %",
    )

    # --- 4-2) 소속기구별 피드백 표 ---

    st.markdown("### 소속기구별 피드백 표")

    df_fb = df_group.copy()

    # 사용 분포 순위 (에너지 사용량 / 공단 전체 에너지 사용량 기준)
    df_fb["사용 분포 순위"] = df_fb["에너지 사용량(현재 기준)"].rank(
        ascending=False, method="dense"
    )

    # 3개년 평균 증가 순위
    df_fb["3개년 평균 증가 순위"] = df_fb["3개년 평균 에너지 사용 증감률"].rank(
        ascending=False, method="dense"
    )

    # 평균 에너지 사용량(연면적 기준) 순위
    df_fb["평균 에너지 사용량(연면적 기준) 순위"] = df_fb[
        "평균 에너지 사용량(연면적 기준) 대비 사용비율"
    ].rank(ascending=False, method="dense")

    # 목표 권장 감축량 (공단 전체 추가 감축 필요량을 기관별 비중으로 배분)
    if baseline_val and baseline_val > 0 and total_U > 0:
        need_total = total_V - baseline_val
        if need_total < 0:
            need_total = 0
        df_fb["목표 권장 감축량"] = need_total * (
            df_fb["에너지 사용량(현재 기준)"] / total_U
        )
    else:
        df_fb["목표 권장 감축량"] = pd.NA

    # 에너지 사용량 증가 사유 제출 대상
    def need_reason(row):
        cond1 = (
            pd.notna(row["3개년 평균 에너지 사용 증감률"])
            and row["3개년 평균 에너지 사용 증감률"] > 0
        )
        cond2 = (
            pd.notna(row["평균 에너지 사용량(연면적 기준) 대비 사용비율"])
            and row["평균 에너지 사용량(연면적 기준) 대비 사용비율"] > 1
        )
        return "O" if (cond1 and cond2) else "X"

    df_fb["에너지 사용량 증가 사유 제출 대상"] = df_fb.apply(need_reason, axis=1)

    # 피드백 표 출력 (전체 폭, NaN → '-')
    st.dataframe(
        df_fb[
            [
                "구분",
                "사용 분포 순위",
                "3개년 평균 증가 순위",
                "평균 에너지 사용량(연면적 기준) 순위",
                "목표 권장 감축량",
                "에너지 사용량 증가 사유 제출 대상",
            ]
        ].style.format(na_rep="-"),
        use_container_width=True,
    )

    # ============================================================
    # 4) 공단 전체 분석·코멘트 (기존 유지)
    # ============================================================

    st.markdown("## 공단 전체 분석·코멘트")

    annual_total_only = analyzer.get_annual_ghg(df_all, by_agency=False)
    actual_emission = annual_total_only.query(
        "연도 == @selected_year"
    )["연간 온실가스 배출량"].sum()

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
        current_month=기준달,
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
