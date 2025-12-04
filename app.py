# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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


# 기관 순서 및 시설군 정의 (공통)
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
# 공통 유틸 (df_std / df_raw 로딩)
# ============================

def load_all_energy_data(base_dir: Path = ENERGY_DIR):
    """저장된 모든 연도 파일을 로드하여
    - 표준 스키마 df_all (연도, 기관명, 월, 온실가스 환산량)
    - 파일 메타 정보
    - 로딩 오류 목록
    을 반환.
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


def get_energy_file_path_for_year(year: int, base_dir: Path = ENERGY_DIR) -> Optional[Path]:
    """파일명에 연도가 포함된 연간 에너지 사용량 파일 경로 탐색."""
    for p in base_dir.glob("*.xlsx"):
        if str(year) in p.name:
            return p
    return None


def load_raw_year_data(year: int) -> pd.DataFrame | None:
    """원본 시트(df_raw)를 로딩 (시트1, U/V/W, 월별 데이터 분석용)."""
    path = get_energy_file_path_for_year(year)
    if path is None:
        return None
    return loader.load_energy_raw_for_analysis(path)


def preprocess_uv_w(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, str, str, str, List[Dict[str, Any]]]:
    """원본 시트의 기관명 + U/V/W 컬럼 정제.

    - 기관명: NaN 제거, 좌우 공백 제거
    - U/V/W: 문자열/공백 처리 후 float 변환
             숫자로 변환 불가한 값은 오류 리스트에 기록하고 NaN 처리
    """
    errors: List[Dict[str, Any]] = []

    # 열 인덱스 기반: C, U, V, W
    org_col = df_raw.columns[2]   # C열 (기관명 계열)
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

        # 완전 공백/빈 문자열 → 결측
        empty_mask = s_str == ""
        s_str = s_str.mask(empty_mask, pd.NA)

        # 숫자로 변환
        converted = pd.to_numeric(s_str, errors="coerce")

        # 변환 실패 로깅(문자열 등)
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
    """월별 열(1월~12월) 중 실제 숫자 데이터가 존재하는 가장 마지막 월 번호."""
    last_month: int | None = None

    # 헤더 기준 월 컬럼 탐지 (예: '1월', '2월')
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
# 탭 구성 (메뉴 구조)
# ============================

tab_dashboard, tab_upload, tab_debug = st.tabs(
    ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그/진단"]
)


# ============================================================
# 📂 1) 에너지 사용량 파일 업로드 탭
# ============================================================

with tab_upload:

    st.header("에너지 사용량 파일 업로드")

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
                    # 파일 저장 + 표준 스키마 변환 (저장은 ENERGY_DIR)
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

    with upload_col2:
        st.markdown("#### 저장된 파일 목록")

        df_all_upload, files_meta, load_errors = load_all_energy_data()

        if files_meta:
            df_files = pd.DataFrame(files_meta).sort_values(
                ["연도", "업로드시간"], ascending=[False, False]
            )
            st.table(df_files[["연도", "파일명", "업로드시간"]])
        else:
            st.info("저장된 파일 없음")


# ============================================================
# 📊 2) 대시보드 탭
#    - 상단 그래프/필터: df_std 기반 (기존 구조 유지)
#    - 에너지 사용량 분석/피드백: df_raw(U/V/W) 기반 전체 재작성
# ============================================================

with tab_dashboard:

    # ------------------------------
    # 데이터 로딩
    # ------------------------------
    df_all, files_meta, load_errors = load_all_energy_data()

    if df_all is None or df_all.empty:
        st.warning("에너지 사용량 데이터가 없습니다. 먼저 [에너지 사용량 파일 업로드] 탭에서 파일을 업로드해 주세요.")
        st.stop()

    # 표준 스키마 집계 (상단 그래프/지표용)
    datasets = analyzer.build_dashboard_datasets(df_all)
    annual_total = datasets["annual_total"]
    annual_by_agency = datasets["annual_by_agency"]
    monthly_total = datasets["monthly_total"]
    monthly_by_agency = datasets["monthly_by_agency"]

    years = sorted(df_all["연도"].dropna().unique().tolist())
    default_year = max(years)

    # ------------------------------
    # 진행 중 기능 안내
    # ------------------------------
    with st.expander("🛠️ 현재 진행 중인 기능 반영 현황"):
        st.markdown(
            """\
            - 상단 에너지 사용량 추이(필터 + 그래프 2개) 레이아웃 유지
            - 기준배출량 기능 전면 제거
            - 에너지 사용량 분석/피드백은 **df_raw(U/V/W)** 기반으로 재작성
            - NaN/None은 0으로 대체하지 않고, 전처리 후 계산 불가 상황만 '-' 표시
            """
        )

    # ========================================================
    # 2-1) 에너지 사용량 추이 (기존 상단 레이아웃 유지)
    # ========================================================

    st.markdown("## 에너지 사용량 추이")

    filter_col, main_col = st.columns([1, 3])

    # -------- 좌측 필터 --------
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

    # -------- 우측 요약 패널 + 그래프 --------
    with main_col:
        # 공단 전체 연간 배출량
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

        # 그래프 데이터 구성
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

    # ========================================================
    # 2-2) 에너지 사용량 분석 (df_raw 기반, 시트1 명세)
    # ========================================================

    st.markdown("---")
    st.markdown("## 에너지 사용량 분석")

    raw_df_original = load_raw_year_data(int(selected_year))
    if raw_df_original is None:
        st.error(f"{selected_year}년 원본 에너지 사용량 파일을 찾을 수 없습니다.")
        st.stop()

    # U/V/W 전처리
    raw_df, org_col, U_col, V_col, W_col, preprocess_errors = preprocess_uv_w(
        raw_df_original
    )

    # ---------- 2-1. 공단 전체 기준 ----------
    total_U = float(raw_df[U_col].sum(skipna=True))
    total_V = float(raw_df[V_col].sum(skipna=True))

    # 최근 3개년 U합계 평균
    past_years = [int(selected_year) - 3, int(selected_year) - 2, int(selected_year) - 1]
    past_u_values: List[float] = []
    for y in past_years:
        df_past_raw = load_raw_year_data(y)
        if df_past_raw is not None:
            df_past, p_org, p_U, p_V, p_W, _ = preprocess_uv_w(df_past_raw)
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

    # W 평균 (의료/복지/기타)
    st.markdown("#### 평균 에너지 사용량(W 평균)")

    def avg_group(names: List[str]) -> Optional[float]:
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

    # ---------- 2-2. 소속기구별 분석 ----------
    st.markdown("### 소속기구별 분석")

    # 기관별 집계: U합, V합, W평균
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

    # 전체 대비 사용량 분포비율 = 기관 U / 전체 U
    if total_U != 0:
        df_group["전체 대비 사용량 분포비율"] = df_group["U_sum"] / total_U * 100
    else:
        df_group["전체 대비 사용량 분포비율"] = pd.NA

    # 평균 에너지 사용량 대비 사용비율 = 기관 W평균 / 시설군 W평균
    def w_ratio(row):
        if row["시설구분"] == "의료시설":
            base = med_avg
        elif row["시설구분"] == "복지시설":
            base = wel_avg
        else:
            base = oth_avg
        if base in (None, 0) or pd.isna(row["W_mean"]):
            return pd.NA
        return row["W_mean"] / base

    df_group["평균 에너지 사용량(W) 대비 사용비율"] = df_group.apply(w_ratio, axis=1)

    # 기관별 3개년 평균 대비 증감률 = (올해U - 과거3개년U평균) / 과거3개년U평균
    def three_year_rate(name: str) -> Optional[float]:
        vals: List[float] = []
        for y in past_years:
            dfp_raw = load_raw_year_data(y)
            if dfp_raw is not None:
                dfp, p_org, p_U, p_V, p_W, _ = preprocess_uv_w(dfp_raw)
                dfp = dfp[dfp[p_org].notna()].copy()
                dfp[p_org] = dfp[p_org].astype(str).str.strip()
                vals.append(float(dfp[dfp[p_org] == name][p_U].sum(skipna=True)))
        if not vals:
            return None
        avg_past = sum(vals) / len(vals)
        if avg_past == 0:
            return None
        now_u = float(df_group.loc[df_group["구분"] == name, "U_sum"].iloc[0])
        return (now_u - avg_past) / avg_past * 100

    df_group["3개년 평균 대비 증감률"] = df_group["구분"].apply(three_year_rate)

    # 표시용 컬럼명 정리
    df_group_display = df_group.rename(columns={
        "U_sum": "에너지 사용량(현재 기준)",
        "V_sum": "면적당 온실가스 배출량",
        "W_mean": "W평균",
    })

    # 기관 순서 고정
    df_group_display["구분"] = pd.Categorical(
        df_group_display["구분"], categories=FACILITY_ORDER, ordered=True
    )
    df_group_display = df_group_display.sort_values("구분")

    cols_analysis = [
        "구분",
        "시설구분",
        "에너지 사용량(현재 기준)",
        "면적당 온실가스 배출량",
        "전체 대비 사용량 분포비율",
        "평균 에너지 사용량(W) 대비 사용비율",
        "3개년 평균 대비 증감률",
    ]

    st.dataframe(
        df_group_display[cols_analysis].style.format(na_rep="-"),
        use_container_width=True,
    )

    # ========================================================
    # 2-3) 피드백 (df_raw 기반, 시트2 명세)
    # ========================================================

    st.markdown("## 피드백")

    # ---------- 3-1. 공단 전체 기준 ----------
    st.markdown("### 공단 전체 기준")

    기준달 = detect_last_month_with_data(raw_df_original)
    f1 = st.columns(1)[0]
    f1.metric("기준 달", f"{기준달}월" if 기준달 is not None else "-")

    # ---------- 3-2. 소속기구별 피드백 ----------
    st.markdown("### 소속기구별 피드백")

    df_fb = df_group_display.copy()

    # 사용량 분포 순위 (U합계 기준)
    df_fb["사용량 분포 순위"] = df_fb["에너지 사용량(현재 기준)"].rank(
        ascending=False, method="dense"
    )

    # 에너지 3개년 평균 증가 순위 (3개년 평균 대비 증감률 기준)
    df_fb["에너지 3개년 평균 증가 순위"] = df_fb["3개년 평균 대비 증감률"].rank(
        ascending=False, method="dense"
    )

    # 평균 에너지 사용량(W) 기준 순위
    df_fb["평균 에너지 사용량(W) 기준 순위"] = df_fb[
        "평균 에너지 사용량(W) 대비 사용비율"
    ].rank(ascending=False, method="dense")

    # 권장 감축량: U증가분 + W초과분 기준
    def recommended_reduction(row) -> Optional[float]:
        name = row["구분"]

        # 기관별 3개년 평균 U
        vals: List[float] = []
        for y in past_years:
            dfp_raw = load_raw_year_data(y)
            if dfp_raw is not None:
                dfp, p_org, p_U, p_V, p_W, _ = preprocess_uv_w(dfp_raw)
                dfp = dfp[dfp[p_org].notna()].copy()
                dfp[p_org] = dfp[p_org].astype(str).str.strip()
                vals.append(float(dfp[dfp[p_org] == name][p_U].sum(skipna=True)))

        avg_u = sum(vals) / len(vals) if vals else None
        current_u = row["에너지 사용량(현재 기준)"]

        # U 증가율
        if avg_u is not None and avg_u > 0:
            delta_u = max(current_u - avg_u, 0)
            u_ratio = delta_u / avg_u
        else:
            u_ratio = 0.0

        # 시설군 평균 대비 W 초과율
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

    # ---------- 3-3. 최종 피드백 문장 (단순 유지용 텍스트) ----------
    st.markdown("### 공단 전체 분석·코멘트")
    # 기존 화면의 하단 문장을 유지하는 취지로, 간단한 요약 문장을 생성
    recent_total_df, recent_years = analyzer.get_recent_years_ghg(
        annual_total, base_year=int(selected_year)
    )
    if not recent_total_df.empty and len(recent_years) >= 2:
        df_recent_sorted = recent_total_df.sort_values("연도")
        first_val = float(df_recent_sorted.iloc[0]["연간 온실가스 배출량"])
        last_val = float(df_recent_sorted.iloc[-1]["연간 온실가스 배출량"])
        if last_val > first_val:
            trend = "증가 추세"
        elif last_val < first_val:
            trend = "감소 추세"
        else:
            trend = "유지 추세"
        st.write(
            f"※ 공단 에너지 사용량 증감 분석 결과, 최근 {len(recent_years)}개년 동안 온실가스 배출량은 "
            f"전반적으로 **{trend}**를 보이고 있습니다. 기관별 피드백을 참고하여 추가적인 절감 방안을 검토해 주시기 바랍니다."
        )
    else:
        st.write(
            "※ 공단 에너지 사용량 증감 분석을 위한 충분한 연도 데이터가 확보되지 않았습니다. "
            "추가 연도 데이터 업로드 후 다시 확인해 주세요."
        )


# ============================================================
# 🔧 3) 디버그 / 진단 탭 (구조 진단 롤백)
# ============================================================

with tab_debug:

    st.header("디버그 / 구조 진단")

    st.markdown("### 엑셀 구조 진단")

    uploaded_debug_file = st.file_uploader(
        "엑셀 구조 진단 파일 업로드 (.xlsx)", type=["xlsx"]
    )

    if uploaded_debug_file:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_debug_file.read())
            tmp_path = Path(tmp.name)

        # 1) 원본 헤더(상위 몇 행) 미리보기
        try:
            df_raw_full = pd.read_excel(tmp_path, sheet_name=0, header=None)
            st.subheader("원본 엑셀 헤더(상위 5행)")
            st.dataframe(df_raw_full.head(), use_container_width=True)
        except Exception as e:
            st.error(f"원본 엑셀 읽기 오류: {e}")
        else:
            # 2) 구조 진단 (소속기관 컬럼 / 월별 컬럼 / 온실가스 컬럼 탐지)
            try:
                res = loader.validate_excel_file(tmp_path)
                st.subheader("구조 진단 결과")
                st.json(res)
            except Exception as e:
                st.error(f"구조 진단 오류: {e}")

            # 3) 표준 스키마 변환 결과 미리보기
            try:
                df_std, year = loader.load_energy_xlsx(tmp_path)
                st.subheader(f"표준 스키마 변환 결과 미리보기 (연도: {year})")
                st.dataframe(df_std.head(), use_container_width=True)
            except Exception as e:
                st.error(f"표준 스키마 변환 오류: {e}")
