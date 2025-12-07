# app.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import streamlit as st

# 원그래프(파이 차트)용 - altair 사용, 없으면 graceful degrade
try:
    import altair as alt
    ALT_AVAILABLE = True
except ImportError:
    ALT_AVAILABLE = False


# ===========================================================
# 내부 모듈 import (오류 발생 시 화면에 표시)
# ===========================================================
try:
    from modules.loader import (
        load_spec,
        load_energy_files,
        get_org_order,
        get_year_to_file,
    )
    from modules.analyzer import (
        build_data_2_usage_analysis,
        build_data_3_feedback,
        compute_facility_feedback,
    )
except Exception as e:  # 모듈 import 에러는 바로 보여주고 중단
    st.error("내부 모듈(import) 중 오류가 발생했습니다. app.py / modules 경로를 확인해 주세요.")
    st.exception(e)
    st.stop()


# ===========================================================
# 경로 / 로그 유틸
# ===========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def log_error(msg: str) -> None:
    st.error(msg)


def log_warning(msg: str) -> None:
    st.warning(msg)


# ===========================================================
# 형식 지정 유틸
# ===========================================================
def format_number(value, rule: Mapping) -> str:
    """formatting_rules.json 의 규칙에 따라 숫자를 문자열로 변환."""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value

    value = float(value)
    style = rule.get("style", "number")
    digits = int(rule.get("digits", 0))
    scale = float(rule.get("scale", 1.0))

    scaled = value * scale

    if style == "percent":
        return f"{scaled:.{digits}f}%"
    if style == "integer_comma":
        return f"{int(round(scaled)):,}"
    if style == "float_comma":
        fmt = f"{{:,.{digits}f}}"
        return fmt.format(scaled)
    return str(value)


def format_table(
    df: pd.DataFrame,
    fmt_rules: Mapping[str, Mapping],
    column_fmt_map: Mapping[str, str],
    default_fmt_name: Optional[str] = None,
) -> pd.DataFrame:
    """테이블에 formatting_rules 적용."""
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
# 원그래프(파이 차트) 유틸
# ===========================================================
def render_pie_from_series(series: pd.Series, title: str, use_abs: bool = False) -> None:
    """기관별 값을 받아 원그래프(Altair)를 그린다.

    - use_abs=True: 음수 가능 지표(증감률 등)에 절대값 적용
    - 색상 팔레트: category20
    - 기관명 정렬: value 내림차순(높은 값 → 낮은 값)
    - 기타 그룹 없음: 모든 소속기구를 그대로 표시
    """
    if not ALT_AVAILABLE:
        st.info(f"'{title}' 원그래프를 표시하려면 altair 패키지가 필요합니다.")
        return

    if series is None or series.empty:
        st.info(f"{title}를(을) 표시할 데이터가 없습니다.")
        return

    s = series.dropna()
    if s.empty:
        st.info(f"{title}를(을) 표시할 데이터가 없습니다.")
        return

    if use_abs:
        s = s.abs()

    s = s[s > 0]
    if s.empty:
        st.info(f"{title}를(을) 표시할 유효한 값이 없습니다.")
        return

    # 값 큰 순으로 정렬
    s = s.sort_values(ascending=False)

    df = s.reset_index()
    df.columns = ["기관명", "value"]

    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="value", type="quantitative", stack=True),
            color=alt.Color(
                field="기관명",
                type="nominal",
                sort=alt.SortField(field="value", order="descending"),
                scale=alt.Scale(scheme="category20"),
            ),
            tooltip=[
                alt.Tooltip("기관명:N", title="기관명"),
                alt.Tooltip("value:Q", title="값", format=",.1f"),
            ],
        )
        .properties(title=title)
    )

    st.altair_chart(chart, use_container_width=True)


# ===========================================================
# 에너지 사용량 추이 그래프 유틸
# ===========================================================
def compute_monthly_usage(df_all: pd.DataFrame, year: int) -> pd.Series:
    """
    df_raw_all 기준으로 월별 에너지 사용량(연단위)을 계산한다.
    - '사용년월' 컬럼이 있으면 그대로 사용
    - 없으면 '월' 컬럼 또는 날짜에서 월을 추출
    """
    if df_all is None or df_all.empty:
        return pd.Series(dtype=float)

    df_year = df_all[df_all["연도"] == year].copy()
    if df_year.empty:
        return pd.Series(dtype=float)

    month_col = None
    for cand in ["사용년월", "월", "month"]:
        if cand in df_year.columns:
            month_col = cand
            break

    if month_col is None:
        # 날짜 형태 컬럼에서 월 파싱 시도
        date_cols = [c for c in df_year.columns if re.search("일자|date", c)]
        for c in date_cols:
            try:
                df_year[c] = pd.to_datetime(df_year[c], errors="coerce")
                if df_year[c].notna().any():
                    df_year["월"] = df_year[c].dt.month
                    month_col = "월"
                    break
            except Exception:
                continue

    if month_col is None:
        return pd.Series(dtype=float)

    month_series = pd.to_numeric(df_year[month_col], errors="coerce")
    df_year = df_year.assign(__월=month_series)
    df_year = df_year[df_year["__월"].between(1, 12)]

    if df_year.empty:
        return pd.Series(dtype=float)

    monthly = df_year.groupby("__월")["연단위"].sum()
    monthly = monthly.reindex(range(1, 13), fill_value=0.0)
    monthly.index.name = "월"
    return monthly


def compute_annual_usage(df_all: pd.DataFrame, years: Mapping[int, pd.DataFrame]) -> pd.Series:
    """df_raw_all 기준으로 연도별 총 사용량(연단위)을 계산."""
    if df_all is None or df_all.empty:
        return pd.Series(dtype=float)

    annual = df_all.groupby("연도")["연단위"].sum()
    all_years = sorted(years.keys())
    annual = annual.reindex(all_years, fill_value=0.0)
    annual.index.name = "연도"
    return annual


def render_usage_trend_charts(
    df_raw_all: pd.DataFrame,
    year_to_raw: Mapping[int, pd.DataFrame],
    selected_year: int,
) -> None:
    """월별/연도별 에너지 사용량 추이 그래프 섹션을 출력."""
    if df_raw_all is None or df_raw_all.empty:
        st.info("에너지 사용량 추이를 표시할 df_raw 데이터가 없습니다.")
        return

    col_month, col_year = st.columns(2)

    with col_month:
        st.markdown("**월별 에너지 사용량 추이**")
        monthly = compute_monthly_usage(df_raw_all, selected_year)
        if monthly.empty:
            st.info("월 정보를 찾을 수 없어 그래프를 표시할 수 없습니다.")
        else:
            chart_data = pd.DataFrame({"월": monthly.index, "에너지 사용량": monthly.values})
            st.line_chart(
                chart_data.set_index("월"),
                use_container_width=True,
            )

    with col_year:
        st.markdown("**연도별 에너지 사용량 추이 (최대 5개년)**")
        annual = compute_annual_usage(df_raw_all, year_to_raw)
        if annual.empty:
            st.info("연도별 에너지 사용량을 계산할 수 없습니다.")
        else:
            if len(annual) > 5:
                annual = annual.sort_index().iloc[-5:]

            chart_data = pd.DataFrame({"연도": annual.index.astype(str), "에너지 사용량": annual.values})
            st.bar_chart(
                chart_data.set_index("연도"),
                use_container_width=True,
            )


# ===========================================================
# 대시보드 탭 렌더링
# ===========================================================
def render_dashboard_tab(
    spec: dict,
    fmt_rules: Mapping[str, Mapping],
    analysis_year_to_raw: Mapping[int, pd.DataFrame],
    selected_year: int,
    view_mode: str,
    selected_org: Optional[str],
    df_raw_all: Optional[pd.DataFrame],
) -> None:
    if not analysis_year_to_raw:
        st.info(
            "선택된 조건에 해당하는 df_raw 데이터가 없습니다. "
            "먼저 '📂 에너지 사용량 파일 업로드' 탭에서 파일을 업로드해 주세요."
        )
        return

    # -------------------------------------------------------
    # 0. 에너지 사용량 추이 (그래프 섹션)
    # -------------------------------------------------------
    st.subheader("에너지 사용량 추이")

    try:
        render_usage_trend_charts(df_raw_all, analysis_year_to_raw, selected_year)
    except Exception as e:
        st.warning("에너지 사용량 추이 그래프를 그리는 중 오류가 발생했습니다.")
        st.exception(e)

    st.markdown("---")

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
        log_error(f"에너지 사용량 분석(Data2) 계산 중 오류가 발생했습니다: {e}")
        st.exception(e)
        return

    data2_overall = data2.overall.copy()
    data2_by_org = data2.by_org.copy()

    DATA2_OVERALL_FMT = {
        "에너지 사용량(현재 기준)": "energy_kwh_int",
        "전년대비 증감률": "percent_2",
        "3개년 평균 에너지 사용량 대비 증감률": "percent_2",
        "의료시설": "percent_2",
        "복지시설": "percent_2",
        "기타시설": "percent_2",
    }
    DATA2_BYORG_FMT = {
        "연면적": "area_m2_int",
        "에너지 사용량": "energy_kwh_int",
        "면적대비 에너지 사용비율": "percent_2",
        "에너지 사용 비중": "percent_2",
        "3개년 평균 에너지 사용량 대비 증감률": "percent_2",
        "시설별 평균 면적 대비 에너지 사용비율": "percent_2",
    }

    # 1) 공단 전체 기준(포맷 적용 전, 시설구분 컬럼 따로 분리)
    fac_cols = ["의료시설", "복지시설", "기타시설"]
    fac_overall = data2_overall[fac_cols].copy()

    # 2) 시설구분별 표용 포맷
    fac_overall_fmt = format_table(
        fac_overall,
        fmt_rules,
        {col: "percent_2" for col in fac_cols},
    )
    fac_overall_fmt = fac_overall_fmt.T
    fac_overall_fmt.columns = ["면적대비 에너지 사용비율"]

    # 3) 공단 전체 기준(시설구분 제외) 포맷
    overall_without_fac = data2_overall.drop(columns=fac_cols, errors="ignore")
    data2_overall_fmt = format_table(
        overall_without_fac,
        fmt_rules,
        DATA2_OVERALL_FMT,
    )

    # 4) 소속기구별 분석
    org_order = list(get_org_order())

    if view_mode == "공단 전체":
        data2_by_org_view = data2_by_org.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        if selected_org in data2_by_org.index:
            data2_by_org_view = data2_by_org.loc[[selected_org]]
        else:
            data2_by_org_view = data2_by_org.iloc[0:0]
    else:
        data2_by_org_view = data2_by_org.reindex(org_order)

    data2_by_org_fmt = format_table(
        data2_by_org_view,
        fmt_rules,
        DATA2_BYORG_FMT,
    )

    col_overall, col_facility = st.columns([2, 1])

    with col_overall:
        st.markdown("**1-1. 공단 전체 기준**")
        st.dataframe(data2_overall_fmt, use_container_width=True)

    with col_facility:
        st.markdown("**1-2. 시설구분별 면적대비 에너지 사용비율**")
        st.dataframe(fac_overall_fmt, use_container_width=True)

    st.markdown("")

    st.markdown("**1-3. 소속기구별 분석(현재 연도 기준)**")
    st.dataframe(data2_by_org_fmt, use_container_width=True)

    # -------------------------------------------------------
    # 1-4. 소속기구별 원그래프(에너지 분석 부문)
    # -------------------------------------------------------
    st.markdown("")

    col_pie_1, col_pie_2 = st.columns(2)
    col_pie_3, col_pie_4 = st.columns(2)
    col_pie_5, col_pie_6 = st.columns(2)
    col_pie_7, col_pie_8 = st.columns(2)

    pie_targets = [
        ("에너지 사용량", False, col_pie_1),
        ("면적대비 에너지 사용비율", False, col_pie_2),
        ("에너지 사용 비중", False, col_pie_3),
        ("3개년 평균 에너지 사용량 대비 증감률", True, col_pie_4),
        ("시설별 평균 면적 대비 에너지 사용비율", False, col_pie_5),
    ]

    for col_name, use_abs, target_col in pie_targets:
        if col_name not in data2_by_org.columns:
            continue
        with target_col:
            st.markdown(f"**{col_name} (소속기구별)**")
            try:
                render_pie_from_series(
                    data2_by_org[col_name].reindex(org_order),
                    title=col_name,
                    use_abs=use_abs,
                )
            except Exception as e:
                st.warning(f"'{col_name}' 원그래프를 표시하는 중 오류가 발생했습니다.")
                st.exception(e)

    st.markdown("---")

    # -------------------------------------------------------
    # 2. 피드백 (data_3)
    # -------------------------------------------------------
    st.subheader("AI 제안 피드백")

    try:
        data3 = build_data_3_feedback(
            analysis_year_to_raw,
            current_year=selected_year,
        )
    except Exception as e:
        log_error(f"피드백(Data3) 계산 중 오류가 발생했습니다: {e}")
        st.exception(e)
        return

    df3_overall = data3.overall.copy()
    df3_by_org = data3.by_org.copy()
    df3_detail = data3.detail.copy()

    DATA3_OVERALL_FMT = {
        "권장 에너지 사용량": "energy_kwh_int",
        "전년대비 감축률": "percent_2",
        "3개년 대비 감축률": "percent_2",
    }
    DATA3_BYORG_FMT = {
        "사용 분포 순위": "integer_comma",
        "에너지 3개년 평균 증가 순위": "integer_comma",
        "평균 에너지 사용량(연면적 기준) 순위": "integer_comma",
        "권장 에너지 사용량": "energy_kwh_int",
        "권장 사용량 대비 에너지 사용 비율": "percent_2",
    }

    df3_overall_fmt = format_table(
        df3_overall,
        fmt_rules,
        DATA3_OVERALL_FMT,
    )

    org_order = list(get_org_order())

    if view_mode == "공단 전체":
        df3_by_org_view = df3_by_org.reindex(org_order)
        df3_detail_view = df3_detail.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        if selected_org in df3_by_org.index:
            df3_by_org_view = df3_by_org.loc[[selected_org]]
        else:
            df3_by_org_view = df3_by_org.iloc[0:0]
        if selected_org in df3_detail.index:
            df3_detail_view = df3_detail.loc[[selected_org]]
        else:
            df3_detail_view = df3_detail.iloc[0:0]
    else:
        df3_by_org_view = df3_by_org.reindex(org_order)
        df3_detail_view = df3_detail.reindex(org_order)

    df3_by_org_fmt = format_table(
        df3_by_org_view,
        fmt_rules,
        DATA3_BYORG_FMT,
    )

    st.markdown("**1. 공단 전체 기준**")
    st.dataframe(df3_overall_fmt, use_container_width=True)
    st.caption("* 온실가스감축목표(NDC) 연평균 감축률 4.17% 기준")

    st.markdown("")
    st.markdown("**2. 소속기구별 권장 사용량 및 관리대상 여부**")
    st.dataframe(df3_by_org_fmt, use_container_width=True)

    st.markdown("")
    st.markdown("**3. 에너지 사용량 관리 대상 상세**")
    st.dataframe(df3_detail_view, use_container_width=True)

    # -------------------------------------------------------
    # 4. 서술형 피드백 (AI 제안 포함)
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("AI 제안 피드백 (서술형)")

    # (1) 종합분석: 간단한 요약 텍스트
    try:
        overall_row = df3_overall.iloc[0]
        cur_usage = data2_overall.iloc[0]["에너지 사용량(현재 기준)"]
        recommended_total = overall_row["권장 에너지 사용량"]
        reduction_vs3 = overall_row["3개년 대비 감축률"]

        high_usage_orgs = (
            df3_by_org.sort_values("사용 분포 순위")
            .head(3)
            .index.tolist()
        )
        high_growth_orgs = (
            df3_by_org.sort_values("에너지 3개년 평균 증가 순위")
            .head(3)
            .index.tolist()
        )

        comment_parts = []

        comment_parts.append(
            f"- 2024년 에너지 사용량은 약 {cur_usage:,.0f}kWh 수준이며, "
            f"권장 사용량 {recommended_total:,.0f}kWh 대비로는 "
            f"{reduction_vs3 * 100:+.2f}% 수준의 감축 여지가 있습니다."
        )

        if high_usage_orgs:
            comment_parts.append(
                f"- 에너지 사용량 비중이 높은 기관은 {', '.join(high_usage_orgs)} 등이며, "
                "이들 기관을 중심으로 절감 대책을 검토하는 것이 효과적입니다."
            )

        if high_growth_orgs:
            comment_parts.append(
                f"- 최근 3개년 평균 대비 사용량 증가 폭이 큰 기관은 {', '.join(high_growth_orgs)} 등으로, "
                "증가 원인(신축ㆍ증축, 장비 교체, 운영시간 증가 등)에 대한 원인 분석이 필요합니다."
            )

        management_targets = df3_by_org[
            df3_by_org["에너지 사용량 관리 대상"] == "O"
        ].index.tolist()
        if management_targets:
            comment_parts.append(
                f"- 종합 조건(면적대비 과사용, 3개년 평균 대비 급증, 권장량 대비 초과)을 고려했을 때 "
                f"우선 관리가 필요한 기관은 {', '.join(management_targets)} 입니다."
            )

        if comment_parts:
            summary_text = "\n".join(f"* {t}" for t in comment_parts)
        else:
            summary_text = "* 피드백을 생성할 수 있는 데이터가 부족합니다."
    except Exception:
        summary_text = "* 종합분석 정보를 불러오는 중 오류가 발생했습니다."

    # (2) 에너지 절감을 위한 제안 (고정 텍스트 – GPT 판단 기반 템플릿)
    ai_suggestion = "\n".join(
        [
            "* 옥상·외벽 등 주요 외피의 단열 성능을 점검하고, 필요 시 단계적으로 보완하여 난방·냉방 부하를 줄입니다.",
            "* 중앙보훈병원, 요양원 등 상시 가동 시설에는 온도·조도·점등을 자동 제어하는 BEMS(건물에너지관리시스템) 도입·확대를 검토합니다.",
            "* 야간·휴일 비상설비 및 대기전력(PC, 복합기, 냉장고 등)을 집중 관리하는 ‘대기전력 차단 캠페인’을 시행합니다.",
            "* 에너지 사용량이 빠르게 증가한 기관을 대상으로 원인 진단(증축, 장비 교체, 운영시간 변경 등)을 실시하고, 기관별 맞춤 절감 목표를 재설정합니다.",
            "* 노후 보일러·냉동기·조명 등 에너지 다소비 설비는 고효율 인증 제품으로 교체하는 중장기 투자계획을 수립합니다.",
            "* 직원 참여형 에너지 절감 프로그램(부서별 절감 실적 공개, 인센티브 부여 등)을 운영하여 자발적 참여를 유도합니다.",
        ]
    )

    st.markdown("**(종합분석)**")
    st.markdown(summary_text)

    st.markdown("")
    st.markdown("**(에너지 절감을 위한 제안)**")
    st.markdown(ai_suggestion)


# ===========================================================
# 📂 업로드 탭 렌더링
# ===========================================================
def render_upload_tab(
    spec: dict,
    year_to_raw: Mapping[int, pd.DataFrame],
) -> None:
    st.subheader("에너지 사용량 파일 업로드")

    st.markdown(
        """
        - 이 탭에서는 연도별 에너지 사용량 원본 파일을 업로드하고, 저장된 파일 목록을 확인할 수 있습니다.
        - 파일 이름에는 반드시 연도가 포함되어야 하며(예: `에너지사용량_2024.xlsx`),
          스펙에 정의된 시트/컬럼 구조를 따라야 합니다.
        """
    )

    col_uploader, col_files = st.columns([2, 1])

    with col_uploader:
        st.markdown("**1. 파일 업로드**")
        uploaded_files = st.file_uploader(
            "엑셀 파일을 선택하세요",
            accept_multiple_files=True,
            type=["xlsx", "xls"],
        )
        if uploaded_files:
            st.info(
                "⚠ 현재 데모 환경에서는 업로드 파일을 영구 저장하지 않고, "
                "세션 동안만 메모리에 보관합니다."
            )

    with col_files:
        st.markdown("**2. 인식된 파일 목록**")
        year_to_file = get_year_to_file()
        if not year_to_file:
            st.info("현재 인식된 에너지 사용량 파일이 없습니다.")
        else:
            rows = []
            for year, path in sorted(year_to_file.items()):
                rows.append({"연도": year, "파일명": Path(path).name})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if year_to_raw:
        st.markdown("---")
        st.markdown("**3. 샘플 df_raw 미리보기 (디버그용)**")
        first_year = sorted(year_to_raw.keys())[0]
        st.caption(f"예시 연도: {first_year}")
        st.dataframe(
            year_to_raw[first_year].head(100),
            use_container_width=True,
        )


# ===========================================================
# 📊 백데이터 분석 탭 (이미 구현되어 있다고 가정 – 요약만 표시)
# ===========================================================
def render_baseline_tab(
    spec: dict,
    year_to_raw: Mapping[int, pd.DataFrame],
    df_raw_all: pd.DataFrame,
) -> None:
    st.subheader("백데이터 분석(요약)")

    if df_raw_all is None or df_raw_all.empty:
        st.info("df_raw 데이터가 없어 백데이터 분석 결과를 표시할 수 없습니다.")
        return

    st.markdown(
        """
        - 이 탭은 기존 백데이터 분석 시트의 주요 지표를 요약해서 보여줍니다.
        - 상세 계산은 baseline.py / analyzer.py 에서 수행되며, 이 화면에서는 결과 일부만 확인합니다.
        """
    )

    years_available = sorted(year_to_raw.keys())
    selected_year = st.selectbox("연도 선택", years_available)

    df_year = df_raw_all[df_raw_all["연도"] == selected_year]
    st.markdown("**선택 연도 df_raw 요약**")
    st.dataframe(df_year.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown("**연도별 3개년 평균 에너지 사용량(요약)**")

    # 실제 baseline 계산 대신, 단순히 연도별 총 사용량을 예시로 표시
    total_by_year = df_raw_all.groupby("연도")["연단위"].sum().sort_index()
    tbl_avg3 = pd.DataFrame(
        {
            "연도": total_by_year.index,
            "총 에너지 사용량": total_by_year.values,
        }
    )

    # 형식 지정
    fmt_rules = spec.get("formatting_rules", {})
    no_format_for_label = {
        "연도": None,
        "총 에너지 사용량": "energy_kwh_int",
    }
    tbl_avg3_fmt = format_table(
        tbl_avg3,
        fmt_rules,
        column_fmt_map=no_format_for_label,
        default_fmt_name="integer_comma",
    )
    st.dataframe(tbl_avg3_fmt, use_container_width=True, hide_index=True)


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

    if df_raw_all is not None and not df_raw_all.empty:
        st.markdown("**df_raw_all 상위 100행**")
        st.dataframe(df_raw_all.head(100), use_container_width=True)
    else:
        st.info("df_raw_all 이 비어 있습니다.")


# ===========================================================
# 메인 엔트리 – 전체 앱 레이아웃
# ===========================================================
def main() -> None:
    st.set_page_config(
        page_title="보훈공단 에너지 사용량 관리 대시보드",
        layout="wide",
    )

    st.title("보훈공단 에너지 사용량 관리 대시보드")

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
    # 1. 에너지 사용량 파일 로딩 (캐시 + 실제 파일 동기화)
    # -------------------------------------------------------
    year_to_raw: Dict[int, pd.DataFrame] = st.session_state.get(
        "year_to_raw_cache", {}
    )
    df_raw_all: Optional[pd.DataFrame] = st.session_state.get("df_raw_all_cache")

    # 현재 인식된 파일 목록
    year_to_file = get_year_to_file()

    # 파일은 있는데 캐시가 없거나(df_raw_all 이 None/empty) 하면 강제 재로딩
    if year_to_file and (not year_to_raw or df_raw_all is None or df_raw_all.empty):
        try:
            year_to_raw, df_raw_all = load_energy_files(year_to_file)
            st.session_state["year_to_raw_cache"] = year_to_raw
            st.session_state["df_raw_all_cache"] = df_raw_all
        except Exception as e:
            st.warning(
                "에너지 사용량 파일을 읽는 중 오류가 발생했습니다. "
                "업로드 탭에서 파일 목록과 형식을 다시 확인해 주세요."
            )
            st.exception(e)
            year_to_raw = {}
            df_raw_all = None
    elif not year_to_file:
        # 파일 자체가 없으면 캐시도 비움
        year_to_raw = {}
        df_raw_all = None

    # -------------------------------------------------------
    # 1-1. 현재 분석 가능한 연도 목록 계산
    # -------------------------------------------------------
    if year_to_raw:
        years_available = sorted(year_to_raw.keys())
    else:
        years_available = []

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
    tab_dashboard, tab_upload, tab_baseline, tab_debug = st.tabs(
        [
            "📊 대시보드",
            "📂 에너지 사용량 파일 업로드",
            "📈 백데이터 분석(요약)",
            "🔧 디버그 / 진단",
        ]
    )

    # 📊 대시보드 탭
    with tab_dashboard:
        if not year_to_raw or df_raw_all is None or selected_year is None:
            st.info("먼저 에너지 사용량 파일을 업로드하고, 사이드바에서 연도를 선택해 주세요.")
        else:
            render_dashboard_tab(
                spec,
                fmt_rules,
                year_to_raw,
                selected_year,
                view_mode,
                selected_org,
                df_raw_all,
            )

    # 📂 업로드 탭
    with tab_upload:
        render_upload_tab(spec, year_to_raw)

    # 📈 백데이터 분석(요약)
    with tab_baseline:
        if not year_to_raw or df_raw_all is None:
            st.info("아직 df_raw 데이터가 없습니다.")
        else:
            render_baseline_tab(spec, year_to_raw, df_raw_all)

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
