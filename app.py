# app.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ===========================================================
# 내부 모듈 import  (오류 발생 시 화면에 표시)
# ===========================================================
try:
    from modules.loader import (
        load_spec,
        load_energy_files,
        get_org_order,
    )
    from modules.analyzer import (
        build_data_2_usage_analysis,
        build_data_3_feedback,
    )
except Exception as e:
    st.error("내부 모듈 import 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# ===========================================================
# 경로
# ===========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

def log_error(msg: str) -> None:
    st.error(msg)

def log_warning(msg: str) -> None:
    st.warning(msg)

# ===========================================================
# 파일명에서 연도 추출
# ===========================================================
def infer_year_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(20[0-9]{2})", name)
    if not m:
        return None
    year = int(m.group(1))
    return year if 2000 <= year <= 2100 else None


# ===========================================================
# data/ 폴더 검색 (로컬 자동 인식)
# ===========================================================
def discover_local_energy_files() -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    if not DATA_DIR.is_dir():
        return mapping

    for p in DATA_DIR.glob("*.xlsx"):
        y = infer_year_from_filename(p.name)
        if y:
            mapping.setdefault(y, p)

    return mapping


# ===========================================================
# 세션 + 로컬 파일 병합
# ===========================================================
def get_year_to_file() -> Dict[int, object]:
    local = discover_local_energy_files()
    session = st.session_state.get("year_to_file", {})

    merged: Dict[int, object] = {}
    merged.update(local)
    merged.update(session)
    return merged

# ===========================================================
# 숫자 포맷팅 (master_energy_spec.formatting_rules 기반)
# ===========================================================
def format_number(value, rule: Mapping) -> str:
    """
    - 천단위 구분기호
    - 소수점 자리수
    - ×100 후 % 등 suffix
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"

    try:
        v = float(value)
    except Exception:
        return str(value)

    # ×100
    if rule.get("multiply_by_100", False):
        v *= 100

    decimals = rule.get("decimal_places", 0)
    thousands = rule.get("thousands_separator", False)
    suffix = rule.get("suffix", "")

    fmt = f"{{:,.{decimals}f}}" if thousands else f"{{:.{decimals}f}}"
    result = fmt.format(v)

    if suffix:
        result += suffix

    return result


# ===========================================================
# DataFrame 포맷팅 적용
# ===========================================================
def format_table(df: pd.DataFrame,
                 fmt_rules: Mapping[str, Mapping],
                 column_fmt_map: Mapping[str, str],
                 default_fmt_name: Optional[str] = None) -> pd.DataFrame:

    if df is None or df.empty:
        return df

    df_fmt = df.copy()

    for col in df_fmt.columns:
        fmt_name = column_fmt_map.get(col, default_fmt_name)
        if not fmt_name:
            continue
        rule = fmt_rules.get(fmt_name)
        if not rule:
            continue
        df_fmt[col] = df_fmt[col].apply(lambda x: format_number(x, rule))

    return df_fmt

# ===========================================================
# data_1 (업로드 탭용) 테이블 생성
# ===========================================================
def build_data1_tables(df_raw_all: pd.DataFrame):
    """
    업로드 탭에서 사용하는 3개 표:
      1) 연도×기관 에너지 사용량(연단위)
      2) 연도×기관 연면적
      3) 연도별 3개년 평균 에너지 사용량 (직전 최대 3개년 평균)
    """
    df = df_raw_all.copy()

    years = sorted(df["연도"].unique())
    org_order = list(get_org_order())

    # 1) 연도×기관 에너지 사용량 (연단위)
    usage = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연단위",      # <- U/V/W 합계 아님. loader에서 제공한 연단위 값 그대로 사용
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=years)
        .reindex(columns=org_order)
    )
    usage["합계"] = usage.sum(axis=1)

    # 2) 연도×기관 연면적
    area = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연면적",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=years)
        .reindex(columns=org_order)
    )
    area["합계"] = area.sum(axis=1)

    # 3) 연도별 3개년 평균 에너지 사용량 (직전 최대 3개년 평균)
    avg3 = pd.DataFrame(index=years, columns=usage.columns, dtype=float)
    for y in years:
        prev_years = [py for py in years if py < y][-3:]
        if not prev_years:
            baseline = usage.loc[y]
        else:
            baseline = usage.loc[prev_years].mean()
        avg3.loc[y] = baseline

    def _reset_index_as_label(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out.insert(0, "구분", out.index.astype(str))
        return out.reset_index(drop=True)

    return (
        _reset_index_as_label(usage),
        _reset_index_as_label(area),
        _reset_index_as_label(avg3),
    )

# ===========================================================
# 📊 대시보드 탭 렌더링 (에너지 사용량 분석 + 피드백)
# ===========================================================
def render_dashboard_tab(
    spec: dict,
    fmt_rules: Mapping[str, Mapping],
    analysis_year_to_raw: Mapping[int, pd.DataFrame],
    selected_year: int,
    view_mode: str,
    selected_org: Optional[str],
) -> None:
    if not analysis_year_to_raw:
        st.info(
            "선택된 조건에 해당하는 df_raw 데이터가 없습니다. "
            "먼저 '📂 에너지 사용량 파일 업로드' 탭에서 파일을 업로드해 주세요."
        )
        return

    # -------------------------------------------------------
    # 1. 에너지 사용량 분석 (data_2)
    # -------------------------------------------------------
    st.subheader("에너지 사용량 분석")

    try:
        data2 = build_data_2_usage_analysis(
            analysis_year_to_raw,
            current_year=selected_year,
        )
    except Exception as e:
        st.exception(e)
        return

    data2_overall = data2.overall.copy()
    data2_by_org = data2.by_org.copy()

    # 기관 정렬 순서 고정
    org_order = list(get_org_order())
    data2_by_org = data2_by_org.reindex(org_order)

    DATA2_OVERALL_FMT = {
        "에너지 사용량(현재 기준)": "energy_kwh_int",
        "전년대비 증감률": "percent_2",
        "3개년 평균 에너지 사용량 대비 증감률": "percent_2",
        "의료시설": "ratio_2",
        "복지시설": "ratio_2",
        "기타시설": "ratio_2",
    }
    DATA2_BYORG_FMT = {
        "연면적": "area_m2_int",
        "에너지 사용량": "energy_kwh_int",
        "면적대비 에너지 사용비율": "ratio_2",
        "에너지 사용 비중": "percent_2",
        "3개년 평균 에너지 사용량 대비 증감률": "percent_2",
        "시설별 평균 면적 대비 에너지 사용비율": "ratio_2",
    }

    df2_overall_fmt = format_table(
        data2_overall,
        fmt_rules,
        DATA2_OVERALL_FMT,
    )
    df2_by_org_fmt = format_table(
        data2_by_org,
        fmt_rules,
        DATA2_BYORG_FMT,
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        suffix = ""
        if view_mode == "기관별" and selected_org:
            suffix = f" ({selected_org})"
        st.markdown(f"**1. 공단 전체 기준{suffix}**")
        st.dataframe(df2_overall_fmt, use_container_width=True)

    with col2:
        st.markdown("**시설구분별 면적대비 평균 에너지 사용비율**")
        fac_cols = ["의료시설", "복지시설", "기타시설"]
        fac_df = df2_overall_fmt[fac_cols].T
        fac_df.columns = ["면적대비 에너지 사용비율"]
        st.dataframe(fac_df, use_container_width=True)

    st.markdown("---")
    st.markdown("**2. 소속기구별 분석**")
    st.dataframe(df2_by_org_fmt, use_container_width=True)

    # -------------------------------------------------------
    # 2. 피드백 (data_3)
    # -------------------------------------------------------
    st.subheader("피드백")

    try:
        data3 = build_data_3_feedback(
            analysis_year_to_raw,
            current_year=selected_year,
        )
    except Exception as e:
        st.exception(e)
        return

    DATA3_OVERALL_FMT = {
        "권장 에너지 사용량": "energy_kwh_int",
        "전년대비 감축률": "percent_2",
        "3개년 대비 감축률": "percent_2",
    }
    DATA3_BYORG_FMT = {
        "권장 에너지 사용량": "energy_kwh_int",
        "권장 사용량 대비 에너지 사용 비율": "percent_2",
    }

    df3_overall_fmt = format_table(
        data3.overall,
        fmt_rules,
        DATA3_OVERALL_FMT,
    )

    df3_by_org = data3.by_org.copy().reindex(org_order)
    df3_by_org_fmt = format_table(
        df3_by_org,
        fmt_rules,
        DATA3_BYORG_FMT,
    )

    st.markdown("**1. 공단 전체 기준**")
    st.dataframe(df3_overall_fmt, use_container_width=True)

    st.markdown("---")
    st.markdown("**2. 소속기구별**")
    st.dataframe(df3_by_org_fmt, use_container_width=True)

    st.markdown("---")
    st.markdown("**3. 에너지 사용량 관리 대상 상세**")
    df3_detail = data3.detail.copy().reindex(org_order)
    st.dataframe(df3_detail, use_container_width=True)

# ===========================================================
# 📂 업로드 탭 렌더링
# ===========================================================
def render_upload_tab(
    spec: dict,
    fmt_rules: Mapping[str, Mapping],
    df_raw_all: Optional[pd.DataFrame],
) -> None:
    st.subheader("공단 에너지 사용량 파일 업로드")

    st.write(
        "- 연도별 《에너지 사용량관리.xlsx》 파일을 업로드하면, "
        "df_raw(U/V/W 기반이 아닌 연단위 기준)로 변환하여 분석에 사용합니다."
    )

    # 1) 파일 업로드 위젯
    uploaded_files = st.file_uploader(
        "연도별 에너지 사용량 파일 업로드 (여러 개 선택 가능)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    # 2) 세션 상태에 업로드 파일 반영
    if uploaded_files:
        year_to_file_session: Dict[int, object] = st.session_state.get(
            "year_to_file", {}
        )
        for f in uploaded_files:
            year = infer_year_from_filename(f.name)
            if year is None:
                log_warning(f"연도를 찾을 수 없어 무시된 파일: {f.name}")
                continue
            year_to_file_session[year] = f
        st.session_state["year_to_file"] = year_to_file_session

    # 3) 현재 인식된 파일 목록 표시
    st.markdown("#### 인식된 연도별 파일 목록")
    merged = get_year_to_file()
    if not merged:
        st.info("현재 인식된 에너지 사용량 파일이 없습니다.")
    else:
        rows = [
            {"연도": year, "파일명": getattr(f, "name", str(f))}
            for year, f in sorted(merged.items())
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # 4) df_raw_all 이 비어 있으면 여기서 한 번 더 로딩을 시도 (안전장치)
    if (df_raw_all is None or df_raw_all.empty) and merged:
        try:
            year_to_raw_tmp, df_raw_all_tmp = load_energy_files(merged)
            df_raw_all = df_raw_all_tmp
            # 디버그용: 행 수 간단 표시
            st.info(f"df_raw가 새로 생성되었습니다. 전체 행 수: {len(df_raw_all)}")
        except Exception as e:
            st.error("df_raw 생성 중 오류가 발생했습니다. 엑셀 형식을 확인해 주세요.")
            st.exception(e)
            return

      # 5) 여전히 df_raw_all 이 없으면 표 생성 불가
    if df_raw_all is None or df_raw_all.empty:
        st.info("아직 df_raw 데이터가 없어 백데이터 분석 표를 생성할 수 없습니다.")
        return

    # 6) data_1용 표 생성
    try:
        tbl_usage, tbl_area, tbl_avg3 = build_data1_tables(df_raw_all)
    except Exception as e:
        st.error("data_1(백데이터 분석) 표 생성 중 오류가 발생했습니다.")
        st.exception(e)
        return

    # 공통: 구분 컬럼은 포맷 적용 안 함 (연도 문자열 그대로 표시)
    no_format_for_label = {"구분": ""}

    # 7) 표 렌더링
    st.markdown("### 1. 연도×기관 에너지 사용량 (연단위)")
    tbl_usage_fmt = format_table(
        tbl_usage,
        fmt_rules,
        column_fmt_map=no_format_for_label,
        # 숫자: 정수, 천단위 콤마, 단위 없음
        default_fmt_name="integer_comma",
    )
    st.dataframe(tbl_usage_fmt, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 2. 연도×기관 연면적")
    tbl_area_fmt = format_table(
        tbl_area,
        fmt_rules,
        column_fmt_map=no_format_for_label,
        default_fmt_name="integer_comma",
    )
    st.dataframe(tbl_area_fmt, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3. 연도별 3개년 평균 에너지 사용량")
    tbl_avg3_fmt = format_table(
        tbl_avg3,
        fmt_rules,
        column_fmt_map=no_format_for_label,
        default_fmt_name="integer_comma",
    )
    st.dataframe(tbl_avg3_fmt, use_container_width=True, hide_index=True)


    st.markdown("---")
    st.markdown("### 2. 연도×기관 연면적")
    tbl_area_fmt = format_table(
        tbl_area,
        fmt_rules,
        column_fmt_map={},
        default_fmt_name="area_m2_int",
    )
    st.dataframe(tbl_area_fmt, use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. 연도별 3개년 평균 에너지 사용량")
    tbl_avg3_fmt = format_table(
        tbl_avg3,
        fmt_rules,
        column_fmt_map={},
        default_fmt_name="energy_kwh_int",
    )
    st.dataframe(tbl_avg3_fmt, use_container_width=True)


# ===========================================================
# 🔧 디버그 / 진단 탭 렌더링
# ===========================================================
def render_debug_tab(
    year_to_raw: Mapping[int, pd.DataFrame],
    df_raw_all: pd.DataFrame,
) -> None:
    st.subheader("df_raw 메타 정보")

    years_available = sorted(year_to_raw.keys())
    st.write("로딩된 연도:", years_available)

    info_rows = []
    for year, df in year_to_raw.items():
        info_rows.append(
            {
                "연도": year,
                "행 수": len(df),
                "기관 수": df["기관명"].nunique(),
            }
        )
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("df_raw 전체 데이터 (상위 100행)")
    st.dataframe(df_raw_all.head(100), use_container_width=True)

    st.markdown("---")
    st.subheader("df_raw 컬럼 정보")
    st.json(
        {
            "columns": df_raw_all.columns.tolist(),
            "dtypes": {c: str(t) for c, t in df_raw_all.dtypes.items()},
        }
    )

# ===========================================================
# 메인 함수 (1) – 초기 설정 및 spec / 데이터 로딩
# ===========================================================
def main() -> None:
    st.set_page_config(
        page_title="공단 에너지 사용량·온실가스 관리 대시보드",
        layout="wide",
    )

    st.title("공단 에너지 사용량·온실가스 관리 대시보드")

    # -------------------------------------------------------
    # 0. spec 로딩
    # -------------------------------------------------------
    try:
        spec = load_spec()
    except Exception as e:
        log_error(f"사양 파일 로딩 중 오류가 발생했습니다: {e}")
        st.stop()

    fmt_rules: Dict[str, Dict] = spec.get("formatting_rules", {})

    # -------------------------------------------------------
    # 1. 에너지 사용량 파일 로딩 (항상 시도)
    # -------------------------------------------------------
    year_to_file = get_year_to_file()
    year_to_raw: Dict[int, pd.DataFrame] = {}
    df_raw_all: Optional[pd.DataFrame] = None

    if year_to_file:
        try:
            year_to_raw, df_raw_all = load_energy_files(year_to_file)
        except Exception as e:
            st.warning(
                "에너지 사용량 파일을 읽는 중 오류가 발생했습니다. "
                "업로드 탭에서 파일을 다시 확인해 주세요."
            )
            st.exception(e)

    years_available = sorted(year_to_raw.keys())

    # -------------------------------------------------------
    # 2. 사이드바 필터
    # -------------------------------------------------------
    with st.sidebar:
        st.header("필터")

        view_mode = st.radio("보기 범위", ["공단 전체", "기관별"], index=0)

        if years_available:
            current_year_spec = int(spec["meta"]["current_year"])
            if current_year_spec in years_available:
                default_year = current_year_spec
            else:
                default_year = years_available[-1]

            selected_year = st.selectbox(
                "이행연도 선택",
                years_available,
                index=years_available.index(default_year),
            )

            df_year = (
                df_raw_all[df_raw_all["연도"] == selected_year]
                if df_raw_all is not None
                else pd.DataFrame()
            )
            orgs_in_data = (
                df_year["기관명"].dropna().unique().tolist()
                if not df_year.empty
                else []
            )

            org_order = list(get_org_order())
            orgs_in_data = sorted(
                [o for o in org_order if o in orgs_in_data],
                key=org_order.index,
            )

            selected_org: Optional[str] = None
            if view_mode == "기관별":
                if not orgs_in_data:
                    log_warning(f"{selected_year}년 데이터에 소속기구가 없습니다.")
                else:
                    selected_org = st.selectbox("소속기구 선택", orgs_in_data)
        else:
            selected_year = None
            selected_org = None
            st.info("아직 분석 가능한 에너지 사용량 데이터가 없습니다.")

        st.selectbox(
            "에너지 종류",
            ["전체"],
            index=0,
            help="현재 버전에서는 전체 에너지 사용량 기준으로 계산합니다.",
        )

    # -------------------------------------------------------
    # 3. 탭 구성
    # -------------------------------------------------------
    tab_dashboard, tab_upload, tab_debug = st.tabs(
        ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
    )

    # 분석에 사용할 year_to_raw (공단/기관별 구분)
    if (
        selected_year is not None
        and view_mode == "기관별"
        and selected_org is not None
        and year_to_raw
    ):
        filtered_year_to_raw: Dict[int, pd.DataFrame] = {}
        for year, df in year_to_raw.items():
            sub = df[df["기관명"] == selected_org].copy()
            if not sub.empty:
                filtered_year_to_raw[year] = sub
        analysis_year_to_raw: Mapping[int, pd.DataFrame] = filtered_year_to_raw
    else:
        analysis_year_to_raw = year_to_raw

    # 📊 대시보드 탭
    with tab_dashboard:
        if selected_year is None:
            st.info(
                "아직 분석 가능한 df_raw 데이터가 없습니다. "
                "먼저 '📂 에너지 사용량 파일 업로드' 탭에서 연도별 파일을 업로드해 주세요."
            )
        else:
            render_dashboard_tab(
                spec,
                fmt_rules,
                analysis_year_to_raw,
                selected_year,
                view_mode,
                selected_org,
            )

    # 📂 업로드 탭
    with tab_upload:
        render_upload_tab(spec, fmt_rules, df_raw_all=df_raw_all)

    # 🔧 디버그 탭
    with tab_debug:
        if not year_to_raw or df_raw_all is None:
            st.info("아직 df_raw 데이터가 없습니다.")
        else:
            render_debug_tab(year_to_raw, df_raw_all)

# ===========================================================
# 엔트리 포인트
# ===========================================================
if __name__ == "__main__":
    main()
