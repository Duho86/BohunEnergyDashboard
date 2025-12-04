# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st

from modules import loader, analyzer


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
    "중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원",
]
WELFARE_FACILITIES = [
    "수원요양원", "광주요양원", "김해요양원", "대구요양원",
    "대전요양원", "남양주요양원", "원주요양원", "전주요양원",
]
OTHER_FACILITIES = [
    "본사", "교육연구원", "보훈원", "재활체육센터", "휴양원",
]


# ============================
# 공통 유틸
# ============================

def load_all_energy_data(base_dir: Path = ENERGY_DIR):
    """저장된 모든 연도 파일을 로드하여
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
    """에너지 사용량관리 엑셀의 원본 구조(시트1)를 그대로 읽어온다."""
    for p in ENERGY_DIR.glob("*.xlsx"):
        if str(year) in p.name:
            return loader.load_energy_raw_for_analysis(p)
    return None


def preprocess_uv_w(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, str, str, str, List[Dict[str, Any]]]:
    """원본 시트의 U/V/W 및 기관명 컬럼을 정제한다.

    - 기관명: 공백 제거, NaN 행 제거
    - U/V/W: float 변환, 변환 실패 값은 오류 리스트에 기록 후 NaN 처리
    - NaN은 집계에서 자동 제외되며, 계산 불가 시 결과를 NaN으로 남긴다.
    """
    errors: List[Dict[str, Any]] = []

    org_col = df_raw.columns[2]   # C열
    U_col = df_raw.columns[20]    # U열
    V_col = df_raw.columns[21]    # V열
    W_col = df_raw.columns[22]    # W열

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
    """월별 열(1월~12월) 중 실제 숫자 데이터가 존재하는 가장 마지막 월을 반환."""
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


# ============================
# 탭 구성
# ============================

tab_dashboard, tab_debug = st.tabs(
    ["📊 대시보드", "🔧 디버그/진단"]
)


# ============================================================
# 📊 1) 대시보드 탭
# ============================================================

with tab_dashboard:

    # -----------------------------
    # 진행중 기능 반영 현황 표시
    # -----------------------------
    with st.expander("🛠️ 현재 진행 중인 기능 반영 현황"):
        st.markdown(
            """\
            # 🔧 기능 반영 현황

            - 상단 에너지 사용량 추이(필터 + 그래프 2개) 레이아웃 유지
            - 기준배출량 기능 전면 제거
            - 에너지 사용량 분석(시트1 기반) 및 피드백(시트2 기반) 로직 보완
            - 모든 계산은 업로드된 에너지 사용량 엑셀의 U/V/W 열 기준
            """
        )

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
    # 상단 그래프/지표용 집계 데이터
    # -----------------------------
    datasets = analyzer.build_dashboard_datasets(df_all)
    annual_total = datasets["annual_total"]
    annual_by_agency = datasets["annual_by_agency"]
    monthly_total = datasets["monthly_total"]
    monthly_by_agency = datasets["monthly_by_agency"]

    years = sorted(df_all["연도"].dropna().unique().tolist())
    default_year = max(years)

    # ============================================================
    # 1) 에너지 사용량 추이 (기존 상단 영역 유지, 기준배출량 제거)
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

    # ----- 우측 요약 패널 + 그래프 -----
    with main_col:
        # 연간 온실가스 배출량(공단 기준)
        annual_row = annual_total[annual_total["연도"] == selected_year]
        if not annual_row.empty:
            total_emission = float(annual_row["연간 온실가스 배출량"].iloc[0])
        else:
            total_emission = None

        # 전년 대비 증감률
        prev_year = int(selected_year) - 1
        prev_row = annual_total[annual_total["연도"] == prev_year]
        if (total_emission is not None) and (not prev_row.empty):
            prev_emission = float(prev_row["연간 온실가스 배출량"].iloc[0])
            if prev_emission != 0:
                yoy_change = (total_emission - prev_emission) / prev_emission * 100
            else:
                yoy_change = None
        else:
            yoy_change = None

        k1, k2, k3 = st.columns(3)
        k1.metric("선택 연도", f"{selected_year}년")
        k2.metric(
            "연간 온실가스 배출량(공단)",
            "-" if total_emission is None else f"{total_emission:,.0f} tCO2eq",
        )
        k3.metric(
            "전년 대비 증감률",
            "-" if yoy_change is None else f"{yoy_change:,.1f} %",
        )

        # 그래프 데이터
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
    # 2) 에너지 사용량 분석 (시트1 구조 기반)
    # ============================================================

    st.markdown("---")
    st.markdown("## 에너지 사용량 분석")

    raw_df_original = load_raw_year_data(int(selected_year))
    if raw_df_original is None:
        st.error(f"{selected_year}년 원본 파일을 찾을 수 없습니다.")
        st.stop()

    raw_df, org_col, U_col, V_col, W_col, preprocess_errors = preprocess_uv_w(
        raw_df_original
    )

    # ---- 3-1) 공단 전체 기준 ----
    total_U = float(raw_df[U_col].sum(skipna=True))
    total_V = float(raw_df[V_col].sum(skipna=True))

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
            past_u_values.append(float(df_past[p_U].sum(skipna=True)))

    if past_u_values:
        past_avg_U = sum(past_u_values) / len(past_u_values)
        if past_avg_U != 0:
            U_change_rate = (total_U - past_avg_U) / past_avg_U * 100
        else:
            U_change_rate = None
    else:
        past_avg_U = None
        U_change_rate = None

    st.markdown("### 공단 전체 기준")

    k1, k2, k3 = st.columns(3)
    k1.metric("에너지 사용량(현재 기준)", f"{total_U:,.0f}")
    k2.metric("면적당 온실가스 배출량", f"{total_V:,.0f}")
    k3.metric(
        "3개년 평균 대비 증감률",
        "-" if U_change_rate is None else f"{U_change_rate:,.1f} %",
    )

    # 평균 에너지 사용량(W 기준)
    st.markdown("#### 평균 에너지 사용량(연면적 W 기준)")

    def avg_group(names: List[str]) -> float | None:
        df_tmp = raw_df[raw_df[org_col].isin(names)]
        if df_tmp.empty:
            return None
        return float(df_tmp[W_col].mean(skipna=True))

    med_avg = avg_group(MEDICAL_FACILITIES)
    wel_avg = avg_group(WELFARE_FACILITIES)
    oth_avg = avg_group(OTHER_FACILITIES)

    g1, g2, g3 = st.columns(3)
    g1.metric("의료시설 평균(W)", "-" if med_avg is None else f"{med_avg:,.1f}")
    g2.metric("복지시설 평균(W)", "-" if wel_avg is None else f"{wel_avg:,.1f}")
    g3.metric("기타시설 평균(W)", "-" if oth_avg is None else f"{oth_avg:,.1f}")

    # ---- 3-2) 소속기구별 분석 ----
    st.markdown("### 소속기구별 분석")

    df_group = (
        raw_df.groupby(org_col)
        .agg(
            U_sum=(U_col, "sum"),
            V_sum=(V_col, "sum"),
            W_mean=(W_col, "mean"),
        )
        .reset_index()
        .rename(columns={org_col: "구분"})
    )

    def facility_type(name: str) -> str:
        if name in MEDICAL_FACILITIES:
            return "의료시설"
        if name in WELFARE_FACILITIES:
            return "복지시설"
        if name in OTHER_FACILITIES:
            return "기타시설"
        return "기타시설"

    df_group["시설구분"] = df_group["구분"].apply(facility_type)

    # 공단 전체 사용량 대비 분포 비율 U(기관)/U(전체)
    df_group["공단 전체 사용량 대비 분포 비율"] = (
        df_group["U_sum"] / total_U * 100 if total_U != 0 else pd.NA
    )

    # 시설군별 W평균 대비 사용비율
    def avg_ratio(row):
        if row["시설구분"] == "의료시설":
            return row["W_mean"] / med_avg if (med_avg not in (None, 0)) else pd.NA
        if row["시설구분"] == "복지시설":
            return row["W_mean"] / wel_avg if (wel_avg not in (None, 0)) else pd.NA
        return row["W_mean"] / oth_avg if (oth_avg not in (None, 0)) else pd.NA

    df_group["평균 에너지 사용량(W) 대비 사용비율"] = df_group.apply(avg_ratio, axis=1)

    # 기관별 3개년 평균 대비 증감률
    def three_year_rate(name: str) -> float | None:
        vals: List[float] = []
        for y in past_years:
            dfp_raw = load_raw_year_data(y)
            if dfp_raw is not None:
                dfp, p_org, p_U, p_V, p_W, err = preprocess_uv_w(dfp_raw)
                dfp = dfp[dfp[p_org].notna()].copy()
                dfp[p_org] = dfp[p_org].astype(str).str.strip()
                vals.append(float(dfp[dfp[p_org] == name][p_U].sum(skipna=True)))

        if vals:
            avg_past = sum(vals) / len(vals)
            now_val = float(
                df_group.loc[df_group["구분"] == name, "U_sum"].iloc[0]
            )
            if avg_past != 0:
                return (now_val - avg_past) / avg_past * 100
        return None

    df_group["3개년 평균 대비 증감률"] = df_group["구분"].apply(three_year_rate)

    # 표 출력용 컬럼 구성 및 정렬
    df_group_display = df_group.copy()
    df_group_display = df_group_display.rename(columns={
        "U_sum": "에너지 사용량(현재 기준)",
        "V_sum": "면적당 온실가스 배출량",
        "W_mean": "W평균",
    })

    df_group_display["구분"] = pd.Categorical(
        df_group_display["구분"], categories=FACILITY_ORDER, ordered=True
    )
    df_group_display = df_group_display.sort_values("구분")

    cols_order = [
        "구분",
        "시설구분",
        "에너지 사용량(현재 기준)",
        "면적당 온실가스 배출량",
        "공단 전체 사용량 대비 분포 비율",
        "평균 에너지 사용량(W) 대비 사용비율",
        "3개년 평균 대비 증감률",
    ]

    st.dataframe(
        df_group_display[cols_order].style.format(na_rep="-"),
        use_container_width=True,
    )

    # ============================================================
    # 3) 피드백 (시트2 구조 기반)
    # ============================================================

    st.markdown("## 피드백")

    # ---- 4-1) 공단 전체 기준 ----
    st.markdown("### 공단 전체 기준")

    기준달 = detect_last_month_with_data(raw_df_original)

    f1 = st.columns(1)[0]
    f1.metric("기준 달", f"{기준달}월" if 기준달 is not None else "-")

    # ---- 4-2) 소속기구별 피드백 ----
    st.markdown("### 소속기구별 피드백")

    df_fb = df_group_display.copy()

    # 사용량 분포 순위 (U 합계 비율 기준)
    df_fb["사용량 분포 순위"] = df_fb["에너지 사용량(현재 기준)"].rank(
        ascending=False, method="dense"
    )

    # 에너지 3개년 평균 증가 순위
    df_fb["에너지 3개년 평균 증가 순위"] = df_fb["3개년 평균 대비 증감률"].rank(
        ascending=False, method="dense"
    )

    # 평균 에너지 사용량(W) 기준 순위
    df_fb["평균 에너지 사용량(W) 기준 순위"] = df_fb[
        "평균 에너지 사용량(W) 대비 사용비율"
    ].rank(ascending=False, method="dense")

    # 권장 감축량: U증가분 + W초과분 기반
    def recommended_reduction(row) -> float | None:
        # 기관별 3개년 평균 U
        name = row["구분"]
        vals: List[float] = []
        for y in past_years:
            dfp_raw = load_raw_year_data(y)
            if dfp_raw is not None:
                dfp, p_org, p_U, p_V, p_W, err = preprocess_uv_w(dfp_raw)
                dfp = dfp[dfp[p_org].notna()].copy()
                dfp[p_org] = dfp[p_org].astype(str).str.strip()
                vals.append(float(dfp[dfp[p_org] == name][p_U].sum(skipna=True)))

        if vals:
            avg_u = sum(vals) / len(vals)
        else:
            avg_u = None

        current_u = row["에너지 사용량(현재 기준)"]

        # U 증가분(양수일 때만)
        if (avg_u is not None) and (avg_u > 0):
            delta_u = max(current_u - avg_u, 0)
            u_ratio = delta_u / avg_u
        else:
            delta_u = 0.0
            u_ratio = 0.0

        # 시설군 평균 대비 W 초과분
        group = row["시설구분"]
        if group == "의료시설":
            base_w = med_avg
        elif group == "복지시설":
            base_w = wel_avg
        else:
            base_w = oth_avg

        w_mean = row["W평균"]
        if base_w not in (None, 0) and pd.notna(w_mean):
            excess_w_ratio = max(w_mean / base_w - 1, 0)
        else:
            excess_w_ratio = 0.0

        # 권장 감축량: 현재 사용량 × (U증가율 + W초과율)
        scale = u_ratio + excess_w_ratio
        if scale <= 0:
            return 0.0
        return float(current_u * scale)

    df_fb["권장 감축량"] = df_fb.apply(recommended_reduction, axis=1)

    # 에너지 사용량 증가 사유 제출 대상
    def need_reason(row) -> str:
        cond1 = pd.notna(row["3개년 평균 대비 증감률"]) and row["3개년 평균 대비 증감률"] > 0
        cond2 = (
            pd.notna(row["평균 에너지 사용량(W) 대비 사용비율"])
            and row["평균 에너지 사용량(W) 대비 사용비율"] > 1
        )
        return "O" if (cond1 or cond2) else "X"

    df_fb["에너지 사용량 증가 사유 제출 대상"] = df_fb.apply(need_reason, axis=1)

    fb_cols = [
        "구분",
        "사용량 분포 순위",
        "에너지 3개년 평균 증가 순위",
        "평균 에너지 사용량(W) 기준 순위",
        "권장 감축량",
        "에너지 사용량 증가 사유 제출 대상",
    ]

    st.dataframe(
        df_fb[fb_cols].style.format(na_rep="-"),
        use_container_width=True,
    )


# ============================================================
# 🔧 2) 디버그 / 진단 탭
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
