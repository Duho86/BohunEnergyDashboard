# app.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from modules.loader import (
    load_spec,
    load_energy_files,
    get_org_order,
)
from modules.analyzer import (
    build_data_2_usage_analysis,
    build_data_3_feedback,
)


# ======================================================================
# 공통 유틸
# ======================================================================


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def log_error(msg: str) -> None:
    st.error(msg)


def log_warning(msg: str) -> None:
    st.warning(msg)


# ======================================================================
# 파일 탐색 / 세션 상태 관리
# ======================================================================


def infer_year_from_filename(name: str) -> Optional[int]:
    """
    파일명에서 연도(20xx)를 추출한다.
    예: '2024년 에너지 사용량관리.xlsx' -> 2024
    """
    m = re.search(r"(20[0-9]{2})", name)
    if not m:
        return None
    year = int(m.group(1))
    if 2000 <= year <= 2100:
        return year
    return None


def discover_local_energy_files() -> Dict[int, Path]:
    """
    data/ 폴더에서 연도 정보를 가진 엑셀 파일을 찾아 {연도: 경로} 매핑을 만든다.
    """
    mapping: Dict[int, Path] = {}
    if not DATA_DIR.is_dir():
        return mapping

    for path in DATA_DIR.glob("*.xlsx"):
        year = infer_year_from_filename(path.name)
        if year is None:
            continue
        # 세션 업로드 파일이 우선이므로, 여기서는 존재하지 않을 때만 설정.
        mapping.setdefault(year, path)

    return mapping


def get_year_to_file() -> Dict[int, object]:
    """
    로컬(data/) + 세션 업로드 파일을 합쳐서 {연도: 파일} 매핑을 반환한다.
    세션에 있는 파일이 로컬 파일보다 우선한다.
    """
    local_mapping = discover_local_energy_files()
    session_mapping: Dict[int, object] = st.session_state.get(
        "year_to_file", {}
    )

    merged: Dict[int, object] = {}
    merged.update(local_mapping)
    merged.update(session_mapping)
    return merged


# ======================================================================
# 포맷팅 유틸 (master_energy_spec.formatting_rules 사용)
# ======================================================================


def format_number(value, rule: Mapping) -> str:
    """
    master_energy_spec.formatting_rules 의 단일 rule 을 적용해 숫자를 문자열로 변환한다.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    multiply = bool(rule.get("multiply_by_100", False))
    if multiply:
        v *= 100.0

    decimals = int(rule.get("decimal_places", 0))
    thousands = bool(rule.get("thousands_separator", False))
    suffix = str(rule.get("suffix", ""))

    if thousands:
        fmt = f"{{:,.{decimals}f}}"
    else:
        fmt = f"{{:.{decimals}f}}"

    s = fmt.format(v)
    if suffix:
        s = f"{s}{suffix}"
    return s


def format_table(
    df: pd.DataFrame,
    fmt_rules: Mapping[str, Mapping],
    column_fmt_map: Mapping[str, str],
    default_fmt_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    DataFrame에 컬럼별 포맷 rule을 적용해 문자열 DataFrame으로 반환.
    """
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

        df_fmt[col] = df_fmt[col].apply(lambda v: format_number(v, rule))

    return df_fmt


# ======================================================================
# data_1. 업로드 탭: 백데이터 분석용 표 생성
# ======================================================================


def build_data1_tables(df_raw_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    업로드 탭에서 사용하는 3개 표를 생성한다.
      1) 연도×기관 에너지 사용량(연단위)
      2) 연도×기관 연면적
      3) 연도별 3개년 평균 에너지 사용량 (직전 최대 3개년 평균)
    """
    df = df_raw_all.copy()
    df["연단위"] = df["U"] + df["W"] + df["V"]

    years = sorted(df["연도"].unique())
    org_order = list(get_org_order())

    # 1) 연도×기관 에너지 사용량
    usage = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연단위",
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
        prev_years = [py for py in years if py < y]
        prev_years = prev_years[-3:]
        if not prev_years:
            baseline = usage.loc[y]
        else:
            baseline = usage.loc[prev_years].mean()
        avg3.loc[y] = baseline

    # 표시 편의를 위해 index를 '구분' 컬럼으로 돌려준다.
    def _reset_index_as_label(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out.insert(0, "구분", out.index.astype(str))
        out = out.reset_index(drop=True)
        return out

    return (
        _reset_index_as_label(usage),
        _reset_index_as_label(area),
        _reset_index_as_label(avg3),
    )


# ======================================================================
# Streamlit UI
# ======================================================================


def main() -> None:
    st.set_page_config(
        page_title="공단 에너지 사용량·온실가스 관리 대시보드",
        layout="wide",
    )

    st.title("공단 에너지 사용량·온실가스 관리 대시보드")

    # ------------------------------------------------------------------
    # 0. spec 로딩
    # ------------------------------------------------------------------
    try:
        spec = load_spec()
    except Exception as e:  # noqa: BLE001
        log_error(f"사양 파일 로딩 중 오류가 발생했습니다: {e}")
        st.stop()

    fmt_rules: Dict[str, Dict] = spec.get("formatting_rules", {})

    # ------------------------------------------------------------------
    # 1. 에너지 사용량 파일 로딩
    # ------------------------------------------------------------------
    year_to_file = get_year_to_file()

    if not year_to_file:
        st.info(
            "에너지 사용량 엑셀 파일이 발견되지 않았습니다. "
            "먼저 '📂 에너지 사용량 파일 업로드' 탭에서 연도별 파일을 업로드해 주세요."
        )
        # 업로드 탭은 그래도 사용할 수 있어야 하므로, 탭 구조는 생성해 둔다.
        tab_dashboard, tab_upload, tab_debug = st.tabs(
            ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
        )
        with tab_upload:
            render_upload_tab(spec, fmt_rules, df_raw_all=None)
        with tab_debug:
            st.write("아직 로딩된 df_raw 데이터가 없습니다.")
        st.stop()

    try:
        year_to_raw, df_raw_all = load_energy_files(year_to_file)
    except Exception as e:  # noqa: BLE001
        st.exception(e)
        st.stop()

    years_available = sorted(year_to_raw.keys())

    # ------------------------------------------------------------------
    # 2. 사이드바 필터
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("필터")

        view_mode = st.radio("보기 범위", ["공단 전체", "기관별"], index=0)

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

        # 선택 연도에 실제 데이터가 있는 기관만 필터 후보로 사용
        df_year = df_raw_all[df_raw_all["연도"] == selected_year]
        orgs_in_data = df_year["기관명"].dropna().unique().tolist()

        # 표준 순서로 정렬
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

        st.selectbox("에너지 종류", ["전체"], index=0, help="현재 버전에서는 전체 에너지 사용량 기준으로 계산합니다.")

    # ------------------------------------------------------------------
    # 3. 탭 구성
    # ------------------------------------------------------------------
    tab_dashboard, tab_upload, tab_debug = st.tabs(
        ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
    )

    # 분석에 사용할 year_to_raw (공단/기관별 구분)
    if view_mode == "기관별" and selected_org is not None:
        filtered_year_to_raw: Dict[int, pd.DataFrame] = {}
        for year, df in year_to_raw.items():
            sub = df[df["기관명"] == selected_org].copy()
            if not sub.empty:
                filtered_year_to_raw[year] = sub
        analysis_year_to_raw: Mapping[int, pd.DataFrame] = filtered_year_to_raw
    else:
        analysis_year_to_raw = year_to_raw

    # ------------------------------------------------------------------
    # 3-1. 📊 대시보드
    # ------------------------------------------------------------------
    with tab_dashboard:
        if not analysis_year_to_raw:
            log_error("선택된 조건에 해당하는 데이터가 없습니다.")
            st.stop()

        st.subheader("에너지 사용량 분석")

        try:
            data2 = build_data_2_usage_analysis(
                analysis_year_to_raw,
                current_year=selected_year,
            )
        except Exception as e:  # noqa: BLE001
            st.exception(e)
            st.stop()

        # === Data2 포맷팅 ===
        data2_overall = data2.overall.copy()
        data2_by_org = data2.by_org.copy()

        # 기관 정렬 고정
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
            title_suffix = ""
            if view_mode == "기관별" and selected_org:
                title_suffix = f" ({selected_org})"
            st.markdown(f"**1. 공단 전체 기준{title_suffix}**")
            st.dataframe(df2_overall_fmt, use_container_width=True)

        with col2:
            st.markdown("**시설구분별 면적대비 평균 에너지 사용비율**")
            # overall 표의 의료/복지/기타 만 따로 재구성
            fac_cols = ["의료시설", "복지시설", "기타시설"]
            fac_df = df2_overall_fmt[fac_cols].T
            fac_df.columns = ["면적대비 에너지 사용비율"]
            st.dataframe(fac_df, use_container_width=True)

        st.markdown("---")
        st.markdown("**2. 소속기구별 분석**")
        st.dataframe(df2_by_org_fmt, use_container_width=True)

        # ------------------------------------------------------------------
        # 피드백
        # ------------------------------------------------------------------
        st.subheader("피드백")

        try:
            data3 = build_data_3_feedback(
                analysis_year_to_raw,
                current_year=selected_year,
            )
        except Exception as e:  # noqa: BLE001
            st.exception(e)
            st.stop()

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

        # 기관 순서 고정
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

    # ------------------------------------------------------------------
    # 3-2. 📂 에너지 사용량 파일 업로드
    # ------------------------------------------------------------------
    with tab_upload:
        render_upload_tab(spec, fmt_rules, df_raw_all=df_raw_all)

    # ------------------------------------------------------------------
    # 3-3. 🔧 디버그 / 진단
    # ------------------------------------------------------------------
    with tab_debug:
        st.subheader("df_raw 메타 정보")

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


# ======================================================================
# 업로드 탭 렌더링
# ======================================================================


def render_upload_tab(
    spec: dict,
    fmt_rules: Mapping[str, Mapping],
    df_raw_all: Optional[pd.DataFrame],
) -> None:
    st.subheader("공단 에너지 사용량 파일 업로드")

    st.write(
        "- 연도별 《에너지 사용량관리.xlsx》 파일을 업로드하면, "
        "df_raw(U/V/W 기반)로 변환하여 분석에 사용합니다."
    )

    uploaded_files = st.file_uploader(
        "연도별 에너지 사용량 파일 업로드 (여러 개 선택 가능)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    # 세션 상태에 업로드 파일 반영
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

    # 현재 인식된 파일 목록 표시
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

    if df_raw_all is None or df_raw_all.empty:
        st.info("아직 df_raw 데이터가 없어 백데이터 분석 표를 생성할 수 없습니다.")
        return

    # data_1용 표 생성
    try:
        tbl_usage, tbl_area, tbl_avg3 = build_data1_tables(df_raw_all)
    except Exception as e:  # noqa: BLE001
        st.exception(e)
        return

    # 포맷팅 규칙: data_1은 값 전체에 공통 포맷을 적용
    st.markdown("### 1. 연도×기관 에너지 사용량 (연단위)")

    tbl_usage_fmt = format_table(
        tbl_usage,
        fmt_rules,
        column_fmt_map={},
        default_fmt_name="energy_kwh_int",
    )
    st.dataframe(tbl_usage_fmt, use_container_width=True)

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


# ======================================================================

if __name__ == "__main__":
    main()
