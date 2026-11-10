# app.py

from __future__ import annotations

import re
import hashlib
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


# -----------------------------------------------------------
# Streamlit rerun helper (버전별 experimental_rerun / rerun 호환)
# -----------------------------------------------------------
def safe_rerun() -> None:
    """Streamlit 버전에 따라 rerun 함수를 안전하게 호출한다."""
    try:
        import streamlit as _st
    except ImportError:
        return

    if hasattr(_st, "rerun"):
        _st.rerun()
    elif hasattr(_st, "experimental_rerun"):
        _st.experimental_rerun()
    else:
        return


# ===========================================================
# 내부 모듈 import (오류 발생 시 화면에 표시)
# ===========================================================
try:
    from modules.loader import (
        load_spec,
        load_energy_files,
        get_org_order,
        CONSUMPTION_FUELS_ORDER,
        RENEWABLE_FUELS_ORDER,
    )
    from modules.analyzer import (
        build_data_2_usage_analysis,
        build_data_3_feedback,
        get_default_energy_settings,
    )
    from modules.feedback import (
        generate_argon_global_feedback_text,
        generate_argon_institution_feedback_text,
        generate_argon_common_recommendations_text,
        ARGON_ENGINE_VERSION,
    )
except Exception as e:
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
    """data/ 폴더에서 연도 정보를 가진 엑셀 파일을 찾아 {연도: Path} 매핑 생성."""
    mapping: Dict[int, Path] = {}
    if not DATA_DIR.is_dir():
        return mapping

    candidates: Dict[int, list[Path]] = {}
    for p in DATA_DIR.glob("*.xlsx"):
        y = infer_year_from_filename(p.name)
        if not y:
            continue
        candidates.setdefault(y, []).append(p)

    for y, paths in candidates.items():
        latest = max(paths, key=lambda pp: pp.stat().st_mtime)
        mapping[y] = latest

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
# 숫자 포맷팅 (master_energy_spec.formatting_rules 기반 - NONE 방지)
# ===========================================================
def format_number(value, rule: Mapping) -> str:
    """spec.formatting_rules 의 규칙을 적용해 숫자를 문자열로 변환 (None/NaN 시 0으로 안전 표시)."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        v = 0.0
    else:
        try:
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                v = 0.0
        except Exception:
            s_val = str(value).strip()
            if s_val.upper() in ("NONE", "NAN", "NULL", "산정대기", "산정 대기", ""):
                return "0"
            return s_val

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
def format_table(
    df: pd.DataFrame,
    fmt_rules: Mapping[str, Mapping],
    column_fmt_map: Mapping[str, str],
    default_fmt_name: Optional[str] = None,
) -> pd.DataFrame:
    """각 컬럼에 지정된 포맷 규칙을 적용해 문자열 테이블로 변환."""
    if df is None or df.empty:
        return df

    df_fmt = df.copy()

    for col in df_fmt.columns:
        fmt_name = column_fmt_map.get(col, default_fmt_name)
        if not fmt_name:
            df_fmt[col] = df_fmt[col].fillna("-").replace({"None": "-", "NONE": "-"})
            continue
        rule = fmt_rules.get(fmt_name)
        if not rule:
            continue
        df_fmt[col] = df_fmt[col].apply(lambda x: format_number(x, rule))

    return df_fmt


# ===========================================================
# 원그래프(파이 차트) 유틸 - Plotly 기반 (내/외부 지시선 라벨 + 1열 전체 범례)
# ===========================================================
def render_pie_from_series(
    series: pd.Series,
    title: str,
    use_abs: bool = False,
    show_rate_in_legend: bool = False,
) -> None:
    """
    기관별 값을 받아 원그래프를 그린다.
    - 조각이 큰 경우 그래프 내부에 소속기구 표시, 작은 경우 지시선(선 표시)과 함께 외부에 표시
    - 우측 범례는 1열(일렬)로 19개 기관이 잘리지 않도록 전체 출력
    """
    if series is None or series.empty:
        st.info(f"{title}를(을) 표시할 데이터가 없습니다.")
        return

    original = pd.to_numeric(series, errors="coerce").dropna()
    if original.empty:
        st.info(f"{title}를(을) 표시할 데이터가 없습니다.")
        return

    if use_abs:
        chart_value = original.abs()
    else:
        chart_value = original.copy()

    valid_mask = chart_value > 0
    original = original.loc[valid_mask]
    chart_value = chart_value.loc[valid_mask]

    if chart_value.empty:
        st.info(f"{title}를(을) 표시할 유효한 값이 없습니다.")
        return

    df = pd.DataFrame(
        {
            "기관명": chart_value.index.astype(str),
            "value": chart_value.values,
            "actual_value": original.values,
        }
    )

    total_value = df["value"].sum()
    df["파이비중"] = (df["value"] / total_value) if total_value > 0 else 0.0
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    if show_rate_in_legend:
        df["범례"] = df.apply(
            lambda row: f"{row['기관명']} ({row['actual_value'] * 100:+.2f}%)",
            axis=1,
        )
        hover_tmpl = (
            "<b>%{customdata[0]}</b><br>"
            "증감률: %{customdata[1]:+.2%}<br>"
            "비중: %{percent}<extra></extra>"
        )
    else:
        df["범례"] = df.apply(
            lambda row: f"{row['기관명']} ({row['파이비중'] * 100:.2f}%)",
            axis=1,
        )
        hover_tmpl = (
            "<b>%{customdata[0]}</b><br>"
            "값: %{customdata[1]:,.2f}<br>"
            "비중: %{percent}<extra></extra>"
        )

    try:
        import plotly.graph_objects as go
        import plotly.express as px

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=df["범례"],
                    values=df["value"],
                    text=df["기관명"],
                    textinfo="text",          # 그래프 위에는 소속기구명 표시
                    textposition="auto",      # 큰 조각은 내부, 작은 조각은 지시선과 함께 외부 자동 표기
                    insidetextorientation="horizontal",
                    automargin=True,          # 외부 지시선 라벨이 잘리지 않도록 여백 자동 확보
                    sort=False,
                    direction="clockwise",
                    marker=dict(colors=px.colors.qualitative.T10 + px.colors.qualitative.Set3),
                    customdata=np.stack((df["기관명"], df["actual_value"]), axis=-1),
                    hovertemplate=hover_tmpl,
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            height=540,  # 19개 기관 범례가 1열로 잘리지 않고 모두 들어가는 높이 확보
            margin=dict(t=55, b=40, l=30, r=190),
            legend=dict(
                title=dict(text="기관명", font=dict(size=12)),
                orientation="v",      # 1열(세로 일렬) 배치
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11),
                tracegroupgap=2,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.info("원그래프 지시선 표기를 위해 plotly 패키지가 필요합니다.")


# ===========================================================
# 📊 대시보드 탭 렌더링 (소비량 / 신재생 생산량 동일 구조·명확한 구분)
# ===========================================================
def render_dashboard_tab(
    spec: dict,
    fmt_rules: Mapping[str, Mapping],
    analysis_year_to_raw: Mapping[int, pd.DataFrame],
    selected_year: int,
    view_mode: str,
    selected_org: Optional[str],
    is_production: bool = False,
) -> None:
    """
    에너지 소비량 대시보드와 신재생에너지 생산량 대시보드 모두
    동일한 인터페이스 배치, 차트(월별/연도별/반기별/분기별/원그래프),
    분석표(data_2), 피드백 표(data_3), 소속기구 순서(ORG_ORDER),
    그리고 ARGON 엔진 기반 AI 제안 피드백 인터페이스로 렌더링하되,
    소비량과 생산량의 명칭·컬럼·피드백이 명확히 구분되도록 표시한다.
    """
    metric_label = "신재생에너지 생산량" if is_production else "에너지 소비량"
    short_label = "생산량" if is_production else "소비량"

    if not analysis_year_to_raw:
        st.info(
            f"선택된 조건에 해당하는 {metric_label} 데이터가 없습니다. "
            "좌측 사이드바 필터를 조정하거나 '📂 에너지 사용량 파일 업로드' 탭에서 파일을 확인해 주세요."
        )
        return

    # ===========================================================
    # 0. 추이 그래프 섹션 (소비량 / 생산량 명확히 구분)
    # ===========================================================
    st.subheader(f"{metric_label} 추이")

    try:
        df_list = [
            df.copy()
            for df in analysis_year_to_raw.values()
            if df is not None and not df.empty
        ]
        df_all = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    except Exception as e:
        st.warning("그래프용 df_raw 병합 중 오류가 발생했습니다.")
        st.exception(e)
        df_all = pd.DataFrame()

    # ===========================================================
    # 월별 / 연도별 그래프
    # ===========================================================
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"**월별 {metric_label} 추이**")
        if df_all.empty or "연도" not in df_all.columns:
            st.info(f"월별 {metric_label} 그래프를 그릴 데이터가 없습니다.")
        else:
            df_year = df_all[df_all["연도"] == selected_year].copy()
            month_info = []
            for c in df_year.columns:
                m = re.search(r"(\d{1,2})\s*월", str(c))
                if m:
                    month_num = int(m.group(1))
                    if 1 <= month_num <= 12:
                        month_info.append((month_num, c))

            if not month_info:
                st.info("1월~12월 관련 컬럼을 찾지 못해 월별 그래프를 표시할 수 없습니다.")
            else:
                month_info.sort(key=lambda x: x[0])
                month_nums = [m for m, _ in month_info]
                month_cols = [c for _, c in month_info]

                for c in month_cols:
                    df_year[c] = pd.to_numeric(df_year[c], errors="coerce").fillna(0.0)

                monthly = df_year[month_cols].sum(axis=0)
                monthly.index = month_nums
                st.line_chart(monthly)

    with col_g2:
        st.markdown(f"**연도별 {metric_label} 추이 (최대 5개년)**")
        if df_all.empty or "연도" not in df_all.columns:
            st.info(f"연도별 {metric_label} 그래프를 그릴 데이터가 없습니다.")
        elif "연단위" not in df_all.columns:
            st.info(f"연단위 컬럼이 없어 연도별 {metric_label}을(를) 계산할 수 없습니다.")
        else:
            yearly = (
                df_all.groupby("연도", dropna=False)["연단위"]
                .sum()
                .sort_index()
                .tail(5)
            )
            if yearly.empty:
                st.info(f"연도별 {metric_label} 합계를 계산할 수 없습니다.")
            else:
                st.bar_chart(yearly)

    # ===========================================================
    # 반기별 / 분기별 추이
    # ===========================================================
    half_rows = []
    quarter_rows = []

    for year in sorted(analysis_year_to_raw.keys()):
        df_period = analysis_year_to_raw.get(year)
        if df_period is None or df_period.empty:
            continue

        month_cols = {}
        for month in range(1, 13):
            exact_col = f"{month}월"
            if exact_col in df_period.columns:
                month_cols[month] = exact_col
                continue
            for col in df_period.columns:
                col_text = str(col).strip()
                if re.fullmatch(rf"{month}\s*월.*", col_text):
                    month_cols[month] = col
                    break

        if len(month_cols) < 12:
            continue

        monthly_totals = {}
        for month in range(1, 13):
            month_col = month_cols[month]
            monthly_totals[month] = float(
                pd.to_numeric(df_period[month_col], errors="coerce").fillna(0).sum()
            )

        first_half = sum(monthly_totals[m] for m in range(1, 7))
        second_half = sum(monthly_totals[m] for m in range(7, 13))
        half_rows.append({"연도": int(year), "상반기": first_half, "하반기": second_half})

        q1 = sum(monthly_totals[m] for m in range(1, 4))
        q2 = sum(monthly_totals[m] for m in range(4, 7))
        q3 = sum(monthly_totals[m] for m in range(7, 10))
        q4 = sum(monthly_totals[m] for m in range(10, 13))
        quarter_rows.append({"연도": int(year), "1분기": q1, "2분기": q2, "3분기": q3, "4분기": q4})

    half_year_df = pd.DataFrame(half_rows, columns=["연도", "상반기", "하반기"])
    quarter_year_df = pd.DataFrame(quarter_rows, columns=["연도", "1분기", "2분기", "3분기", "4분기"])

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.markdown(f"**반기별 {metric_label} 추이 (연도별 비교)**")
        if half_year_df.empty:
            st.info(f"반기별 {metric_label}을(를) 계산할 데이터가 없습니다.")
        elif not ALT_AVAILABLE:
            st.info("반기별 차트를 표시하려면 altair 패키지가 필요합니다.")
        else:
            half_chart_data = half_year_df.melt(
                id_vars=["연도"],
                value_vars=["상반기", "하반기"],
                var_name="반기",
                value_name=metric_label,
            )
            half_chart_data["누적순서"] = half_chart_data["반기"].map({"상반기": 1, "하반기": 2})
            half_chart_data["연도"] = half_chart_data["연도"].astype(int).astype(str)

            half_chart = (
                alt.Chart(half_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("연도:N", title=None, sort=None, axis=alt.Axis(labelAngle=-90)),
                    y=alt.Y(f"{metric_label}:Q", title=None, stack="zero", axis=alt.Axis(format=",.0f")),
                    color=alt.Color(
                        "반기:N",
                        title=None,
                        scale=alt.Scale(domain=["상반기", "하반기"]),
                        legend=alt.Legend(orient="bottom", direction="horizontal"),
                    ),
                    order=alt.Order("누적순서:Q", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("연도:N", title="연도"),
                        alt.Tooltip("반기:N", title="구분"),
                        alt.Tooltip(f"{metric_label}:Q", title=metric_label, format=",.0f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(half_chart, use_container_width=True)

    with col_g4:
        st.markdown(f"**분기별 {metric_label} 추이 (연도별 비교)**")
        if quarter_year_df.empty:
            st.info(f"분기별 {metric_label}을(를) 계산할 데이터가 없습니다.")
        elif not ALT_AVAILABLE:
            st.info("분기별 차트를 표시하려면 altair 패키지가 필요합니다.")
        else:
            quarter_chart_data = quarter_year_df.melt(
                id_vars=["연도"],
                value_vars=["1분기", "2분기", "3분기", "4분기"],
                var_name="분기",
                value_name=metric_label,
            )
            quarter_chart_data["누적순서"] = quarter_chart_data["분기"].map(
                {"1분기": 1, "2분기": 2, "3분기": 3, "4분기": 4}
            )
            quarter_chart_data["연도"] = quarter_chart_data["연도"].astype(int).astype(str)

            quarter_chart = (
                alt.Chart(quarter_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("연도:N", title=None, sort=None, axis=alt.Axis(labelAngle=-90)),
                    y=alt.Y(f"{metric_label}:Q", title=None, stack="zero", axis=alt.Axis(format=",.0f")),
                    color=alt.Color(
                        "분기:N",
                        title=None,
                        scale=alt.Scale(domain=["1분기", "2분기", "3분기", "4분기"]),
                        legend=alt.Legend(orient="bottom", direction="horizontal"),
                    ),
                    order=alt.Order("누적순서:Q", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("연도:N", title="연도"),
                        alt.Tooltip("분기:N", title="구분"),
                        alt.Tooltip(f"{metric_label}:Q", title=metric_label, format=",.0f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(quarter_chart, use_container_width=True)

    # ===========================================================
    # 현재 선택연도의 실제 입력기간 확인 (동일기간 비교)
    # ===========================================================
    comparison_year_to_raw: Dict[int, pd.DataFrame] = {}
    comparison_end_month = 12
    comparison_period_label = "연간"

    current_period_df = analysis_year_to_raw.get(selected_year)
    if current_period_df is not None and not current_period_df.empty:
        active_months = []
        for month in range(1, 13):
            month_col = f"{month}월"
            if month_col not in current_period_df.columns:
                continue
            month_values = pd.to_numeric(current_period_df[month_col], errors="coerce")
            if month_values.fillna(0).sum() > 0:
                active_months.append(month)
        if active_months:
            comparison_end_month = max(active_months)

    if comparison_end_month == 3:
        comparison_period_label = "1분기"
    elif comparison_end_month == 6:
        comparison_period_label = "상반기"
    elif comparison_end_month == 9:
        comparison_period_label = "1~3분기"
    elif comparison_end_month == 12:
        comparison_period_label = "연간"
    else:
        comparison_period_label = f"1~{comparison_end_month}월 누계"

    for year, source_df in analysis_year_to_raw.items():
        if source_df is None or source_df.empty:
            continue
        df_compare = source_df.copy()
        period_month_cols = [
            f"{m}월" for m in range(1, comparison_end_month + 1) if f"{m}월" in df_compare.columns
        ]
        if not period_month_cols:
            continue
        period_values = df_compare[period_month_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df_compare["연단위"] = period_values.sum(axis=1)
        comparison_year_to_raw[int(year)] = df_compare

    # ===========================================================
    # 1. 에너지 소비량 / 신재생 생산량 분석 (data_2)
    # ===========================================================
    st.subheader(f"{metric_label} 분석")

    if comparison_end_month < 12:
        st.caption(
            f"* {selected_year}년 데이터가 {comparison_period_label}까지 입력되어 있어, "
            f"전년 및 최근 3개년 비교는 각 연도의 동일기간({comparison_period_label}) 기준으로 계산합니다."
        )

    try:
        data2 = build_data_2_usage_analysis(
            comparison_year_to_raw,
            current_year=selected_year,
        )
    except Exception as e:
        st.error(f"{metric_label} 분석(data_2) 계산 중 오류가 발생했습니다.")
        st.exception(e)
        return

    data2_overall = data2.overall.copy()
    data2_by_org = data2.by_org.copy()
    org_order = list(get_org_order())

    if view_mode == "공단 전체":
        data2_by_org = data2_by_org.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        if selected_org in data2_by_org.index:
            data2_by_org = data2_by_org.loc[[selected_org]]
        else:
            data2_by_org = data2_by_org.iloc[0:0]

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

    fac_cols = ["의료시설", "복지시설", "기타시설"]
    fac_overall = data2_overall[fac_cols].copy()
    fac_overall_fmt = format_table(
        fac_overall,
        fmt_rules,
        {col: "percent_2" for col in fac_cols},
    ).T
    fac_overall_fmt.columns = [f"면적대비 {short_label} 비율"]

    df2_overall_fmt = format_table(data2_overall, fmt_rules, DATA2_OVERALL_FMT)
    for col in fac_cols:
        if col in df2_overall_fmt.columns:
            df2_overall_fmt = df2_overall_fmt.drop(columns=[col])

    df2_by_org_fmt = format_table(data2_by_org, fmt_rules, DATA2_BYORG_FMT)

    # 소비량 / 생산량 대시보드별 표 컬럼명 구분 표시
    if is_production:
        df2_overall_fmt = df2_overall_fmt.rename(columns={
            "에너지 사용량(현재 기준)": "신재생 생산량(현재 기준)",
            "3개년 평균 에너지 사용량 대비 증감률": "3개년 평균 생산량 대비 증감률",
        })
        df2_by_org_fmt = df2_by_org_fmt.rename(columns={
            "에너지 사용량": "신재생 생산량",
            "면적대비 에너지 사용비율": "면적대비 생산비율",
            "에너지 사용 비중": "신재생 생산 비중",
            "3개년 평균 에너지 사용량 대비 증감률": "3개년 평균 생산량 대비 증감률",
            "시설별 평균 면적 대비 에너지 사용비율": "시설별 평균 면적 대비 생산비율",
        })
    else:
        df2_overall_fmt = df2_overall_fmt.rename(columns={
            "에너지 사용량(현재 기준)": "에너지 소비량(현재 기준)",
            "3개년 평균 에너지 사용량 대비 증감률": "3개년 평균 소비량 대비 증감률",
        })
        df2_by_org_fmt = df2_by_org_fmt.rename(columns={
            "에너지 사용량": "에너지 소비량",
            "면적대비 에너지 사용비율": "면적대비 소비비율",
            "에너지 사용 비중": "에너지 소비 비중",
            "3개년 평균 에너지 사용량 대비 증감률": "3개년 평균 소비량 대비 증감률",
            "시설별 평균 면적 대비 에너지 사용비율": "시설별 평균 면적 대비 소비비율",
        })

    col1, col2 = st.columns([1.3, 1])
    with col1:
        suffix = f" ({selected_org})" if (view_mode == "기관별" and selected_org) else ""
        st.markdown(f"**1. 공단 전체 기준{suffix}**")
        st.dataframe(df2_overall_fmt, use_container_width=True)

    with col2:
        st.markdown(f"**시설구분별 면적대비 평균 {short_label} 비율**")
        if fac_overall_fmt is not None and not fac_overall_fmt.empty:
            st.dataframe(fac_overall_fmt, use_container_width=True)
        else:
            st.info("시설구분별 데이터가 없습니다.")

    st.markdown("---")

    # ===========================================================
    # 1-1. 소속기구별 원그래프 (소비량 / 생산량 구분 표시)
    # ===========================================================
    st.markdown(f"**소속기구별 원그래프 ({metric_label} 분석 부문)**")

    if data2_by_org is None or data2_by_org.empty or len(data2_by_org.index) < 2:
        st.info("소속기구별 비교를 위한 데이터가 2개 미만입니다.")
    else:
        pie_metrics = [
            (f"{metric_label}", "에너지 사용량", False),
            (f"면적대비 {short_label} 비율", "면적대비 에너지 사용비율", False),
            (f"{short_label} 비중", "에너지 사용 비중", False),
            (f"3개년 평균 {short_label} 대비 증감률", "3개년 평균 에너지 사용량 대비 증감률", True),
            (f"시설별 평균 면적 대비 {short_label} 비율", "시설별 평균 면적 대비 에너지 사용비율", False),
        ]

        for i in range(0, len(pie_metrics), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(pie_metrics):
                    break
                title_kor, col_name, use_abs = pie_metrics[i + j]
                with cols[j]:
                    if col_name in data2_by_org.columns:
                        series = data2_by_org[col_name]
                        show_rate = col_name == "3개년 평균 에너지 사용량 대비 증감률"
                        render_pie_from_series(
                            series,
                            title_kor,
                            use_abs=use_abs,
                            show_rate_in_legend=show_rate,
                        )
                    else:
                        st.info(f"'{col_name}' 컬럼이 없어 원그래프를 표시할 수 없습니다.")

    st.markdown("---")
    st.dataframe(df2_by_org_fmt, use_container_width=True)

    # ===========================================================
    # 2. 피드백 (data_3) - 소비량 및 생산량 동일 인터페이스·순서·명칭 구분
    # ===========================================================
    st.subheader(f"{metric_label} 피드백")

    try:
        data3 = build_data_3_feedback(
            comparison_year_to_raw,
            current_year=selected_year,
        )
    except Exception as e:
        st.error(f"{metric_label} 피드백(data_3) 계산 중 오류가 발생했습니다.")
        st.exception(e)
        return

    # 실제 감축률/증감률 보정
    if (
        data3.overall is not None
        and not data3.overall.empty
        and data2_overall is not None
        and not data2_overall.empty
    ):
        if "전년대비 증감률" in data2_overall.columns:
            yoy_change = pd.to_numeric(
                pd.Series([data2_overall.iloc[0].get("전년대비 증감률", 0.0)]),
                errors="coerce",
            ).iloc[0]
            if not pd.isna(yoy_change):
                data3.overall.loc[data3.overall.index[0], "전년대비 감축률"] = (
                    float(yoy_change) if is_production else -float(yoy_change)
                )

        if "3개년 평균 에너지 사용량 대비 증감률" in data2_overall.columns:
            avg3_change = pd.to_numeric(
                pd.Series([data2_overall.iloc[0].get("3개년 평균 에너지 사용량 대비 증감률", 0.0)]),
                errors="coerce",
            ).iloc[0]
            if not pd.isna(avg3_change):
                data3.overall.loc[data3.overall.index[0], "3개년 대비 감축률"] = (
                    float(avg3_change) if is_production else -float(avg3_change)
                )

    DATA3_OVERALL_FMT = {
        "권장 에너지 사용량": "energy_kwh_int",
        "전년대비 감축률": "percent_2",
        "3개년 대비 감축률": "percent_2",
    }

    DATA3_BYORG_FMT = {
        "권장 에너지 사용량": "energy_kwh_int",
        "권장 사용량 대비 에너지 사용 비율": "percent_2",
    }

    df3_overall_fmt = format_table(data3.overall, fmt_rules, DATA3_OVERALL_FMT)
    df3_by_org = data3.by_org.copy()
    df3_detail = data3.detail.copy()

    # 소속기구 순서(ORG_ORDER) 완벽 동일 정렬
    if view_mode == "공단 전체":
        df3_by_org = df3_by_org.reindex(org_order)
        df3_detail = df3_detail.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        df3_by_org = (
            df3_by_org.loc[[selected_org]]
            if selected_org in df3_by_org.index
            else df3_by_org.iloc[0:0]
        )
        df3_detail = (
            df3_detail.loc[[selected_org]]
            if selected_org in df3_detail.index
            else df3_detail.iloc[0:0]
        )

    df3_by_org_fmt = format_table(df3_by_org, fmt_rules, DATA3_BYORG_FMT)

    if is_production:
        df3_overall_fmt = df3_overall_fmt.rename(columns={
            "권장 에너지 사용량": "목표 신재생 생산량",
            "전년대비 감축률": "전년대비 발전 증감률",
            "3개년 대비 감축률": "3개년 대비 발전 증감률",
        })
        df3_by_org_fmt = df3_by_org_fmt.rename(columns={
            "사용 분포 순위": "생산 분포 순위",
            "에너지 3개년 평균 증가 순위": "신재생 3개년 평균 증가 순위",
            "평균 에너지 사용량(연면적 기준) 순위": "평균 생산량(연면적 기준) 순위",
            "권장 에너지 사용량": "목표 신재생 생산량",
            "권장 사용량 대비 에너지 사용 비율": "목표량 대비 생산 달성률",
            "에너지 사용량 관리 대상": "신재생 설비 점검 대상",
        })
        df3_detail = df3_detail.rename(columns={
            "면적대비 에너지 과사용": "면적대비 발전 비중 상위",
            "에너지 사용량 급증(3개년 평균대비)": "신재생 발전량 변동(3개년 대비)",
            "권장량 대비 에너지 사용량 매우 초과": "목표 발전량 대비 편차 발생",
        })
    else:
        df3_overall_fmt = df3_overall_fmt.rename(columns={
            "권장 에너지 사용량": "권장 에너지 소비량",
        })
        df3_by_org_fmt = df3_by_org_fmt.rename(columns={
            "사용 분포 순위": "소비 분포 순위",
            "에너지 3개년 평균 증가 순위": "소비량 3개년 평균 증가 순위",
            "평균 에너지 사용량(연면적 기준) 순위": "평균 소비량(연면적 기준) 순위",
            "권장 에너지 사용량": "권장 에너지 소비량",
            "권장 사용량 대비 에너지 사용 비율": "권장량 대비 소비 비율",
            "에너지 사용량 관리 대상": "에너지 소비량 관리 대상",
        })

    st.markdown("**1. 공단 전체 기준**")
    st.dataframe(df3_overall_fmt, use_container_width=True)
    if is_production:
        st.caption("* 과거 3개년 평균 발전 실적 및 K-RE100 목표 연계 기준")
    else:
        st.caption("* 온실가스감축목표(NDC) 연평균 감축률 4.17% 기준")

    if comparison_end_month < 12:
        st.caption(
            f"* 전년대비 및 3개년 대비 지표는 각 연도의 동일기간({comparison_period_label}) 실적을 기준으로 계산합니다."
        )

    st.markdown("---")
    st.markdown("**2. 소속기구별**")

    if df3_by_org is None or df3_by_org.empty or len(df3_by_org.index) < 2:
        st.info("순위 비교를 위한 데이터가 2개 미만입니다.")
    else:
        st.markdown(f"**소속기구별 원그래프 ({short_label} 분포 순위)**")
        rank_metrics = [
            (f"{short_label} 3개년 평균 증가 순위", "에너지 3개년 평균 증가 순위"),
            (f"평균 {short_label}(연면적 기준) 순위", "평균 에너지 사용량(연면적 기준) 순위"),
        ]
        cols = st.columns(2)
        for idx, (title_kor, col_name) in enumerate(rank_metrics):
            with cols[idx]:
                if col_name in df3_by_org.columns:
                    rank_series = pd.to_numeric(df3_by_org[col_name], errors="coerce").dropna()
                    rank_series = rank_series[rank_series > 0]
                    if rank_series.empty:
                        st.info(f"'{title_kor}' 데이터가 없습니다.")
                    else:
                        max_rank = rank_series.max()
                        score = max_rank + 1 - rank_series
                        render_pie_from_series(score, title_kor, use_abs=False)
                else:
                    st.info(f"'{title_kor}' 컬럼이 없어 원그래프를 표시할 수 없습니다.")

    st.dataframe(df3_by_org_fmt, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**3. {metric_label} {'설비 점검' if is_production else '관리'} 대상 상세**")

    if df3_detail is None or df3_detail.empty:
        st.info("상세 데이터를 생성할 수 없습니다.")
    else:
        st.dataframe(df3_detail, use_container_width=True)

    # ===========================================================
    # 3. AI 제안 피드백 (소비량 / 생산량 동일 인터페이스 + ARGON 엔진 고도화)
    # ===========================================================
    st.markdown("---")
    st.subheader(f"AI 제안 피드백 ({metric_label})")
    st.caption(f"⚙️ 구동 엔진: **ASTRA/ARGON** (환산계수 및 다중 지표 자동 추론 활성화)")

    if comparison_end_month < 12:
        st.info(
            f"{selected_year}년 데이터는 **{comparison_period_label}**까지 입력되어 있습니다. "
            f"AI 제안 피드백은 연간 총량이 아닌 **각 연도의 동일기간({comparison_period_label}) {short_label}을 기준으로 비교 분석**합니다."
        )
    else:
        st.caption(f"{selected_year}년 전체 연도 {metric_label} 데이터를 기준으로 분석합니다.")

    ai_data2_overall = data2_overall.copy()
    ai_data2_by_org = data2_by_org.copy()
    ai_df3_by_org = df3_by_org.copy()

    if view_mode == "공단 전체":
        ai_data2_by_org = ai_data2_by_org.reindex(org_order)
        ai_df3_by_org = ai_df3_by_org.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        ai_data2_by_org = (
            ai_data2_by_org.loc[[selected_org]]
            if selected_org in ai_data2_by_org.index
            else ai_data2_by_org.iloc[0:0]
        )
        ai_df3_by_org = (
            ai_df3_by_org.loc[[selected_org]]
            if selected_org in ai_df3_by_org.index
            else ai_df3_by_org.iloc[0:0]
        )

    # (종합분석)
    st.markdown(f"#### (종합분석 - {metric_label})")
    global_text = generate_argon_global_feedback_text(
        selected_year=selected_year,
        df3_overall=data3.overall,
        data2_overall=ai_data2_overall,
        data2_by_org=ai_data2_by_org,
        df3_by_org=ai_df3_by_org,
        period_label=comparison_period_label,
        is_production=is_production,
    )
    st.markdown(global_text)

    # [소속기구별 맞춤형 피드백] - 소속기구 순서(ORG_ORDER) 완벽 동일
    st.markdown("---")
    st.markdown(f"#### [소속기구별 맞춤형 피드백 - {metric_label}]")

    if ai_data2_by_org is None or ai_data2_by_org.empty:
        st.info("소속기구별 피드백을 생성할 데이터가 없습니다.")
    else:
        upa_series = (
            pd.to_numeric(ai_data2_by_org["면적대비 에너지 사용비율"], errors="coerce")
            if "면적대비 에너지 사용비율" in ai_data2_by_org.columns
            else pd.Series(dtype=float)
        )
        upa_valid = upa_series[upa_series > 0]
        upa_mean = float(upa_valid.mean()) if not upa_valid.empty else 0.0
        total_orgs = len(ai_data2_by_org)

        for org in ai_data2_by_org.index:
            if org not in ai_df3_by_org.index:
                continue

            row2 = ai_data2_by_org.loc[org]
            row3 = ai_df3_by_org.loc[org]

            if row2.isna().all() and row3.isna().all():
                continue

            inst_text = generate_argon_institution_feedback_text(
                org_name=org,
                row2=row2,
                row3=row3,
                upa_mean=upa_mean,
                total_orgs=total_orgs,
                is_production=is_production,
            )
            st.markdown(inst_text)
            st.markdown("")

    # [공단 공통 제안]
    st.markdown("---")
    if is_production:
        st.markdown("#### [신재생에너지 발전 효율 향상을 위한 공단 공통 제안]")
    else:
        st.markdown("#### [에너지 절감을 위한 공단 공통 제안]")
    st.markdown(
        generate_argon_common_recommendations_text(
            df3_by_org=ai_df3_by_org,
            data2_by_org=ai_data2_by_org,
            is_production=is_production,
        )
    )


# ===========================================================
# 📂 업로드 탭 렌더링
# ===========================================================
def _compute_upload_signature(uploaded_files) -> Optional[str]:
    if not uploaded_files:
        return None
    parts = []
    for f in uploaded_files:
        data = f.getvalue()
        h = hashlib.md5(data).hexdigest()
        parts.append(f"{f.name}:{len(data)}:{h}")
    parts.sort()
    joined = "|".join(parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def render_upload_tab(
    spec: dict,
    fmt_rules: Mapping[str, Mapping],
    df_raw_all: Optional[pd.DataFrame],
) -> None:
    st.subheader("공단 에너지 사용량 파일 업로드")
    st.write(
        "- 연도별 《에너지 사용량관리.xlsx》 파일을 업로드하면, "
        "df_raw(연단위 및 월별 기준)로 변환하여 분석에 사용합니다."
    )

    uploaded_files = st.file_uploader(
        "연도별 에너지 사용량 파일 업로드 (여러 개 선택 가능)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    current_sig = _compute_upload_signature(uploaded_files) if uploaded_files else None
    last_sig = st.session_state.get("last_upload_sig")

    if uploaded_files and current_sig and current_sig != last_sig:
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)

        from modules.loader import discover_local_energy_files, get_year_to_file, load_energy_files

        local_before = discover_local_energy_files()
        before_years = set(local_before.keys())

        new_years: list[int] = []
        updated_years: list[int] = []

        for f in uploaded_files:
            year = infer_year_from_filename(f.name)
            if year is None:
                st.warning(f"연도를 찾을 수 없어 무시된 파일: {f.name}")
                continue

            target_path = DATA_DIR / f"energy_{year}.xlsx"
            if year in before_years:
                updated_years.append(year)
            else:
                new_years.append(year)

            with open(target_path, "wb") as out:
                out.write(f.getvalue())

        st.session_state["year_to_file"] = {}

        merged = get_year_to_file()
        if merged:
            try:
                year_to_raw_tmp, df_raw_all_tmp = load_energy_files(merged)
                st.session_state["year_to_raw_all_v4"] = year_to_raw_tmp
                st.session_state["df_raw_all_v4"] = df_raw_all_tmp
                df_raw_all = df_raw_all_tmp

                if new_years:
                    years_str = ", ".join(str(y) for y in sorted(set(new_years)))
                    st.success(f"새 연도 파일 등록 완료: {years_str}년")
                if updated_years:
                    years_str = ", ".join(str(y) for y in sorted(set(updated_years)))
                    st.success(f"기존 연도 파일 업데이트 완료: {years_str}년")

                st.session_state["last_upload_sig"] = current_sig
            except Exception as e:
                st.error("에너지 사용량 파일 로딩 중 오류가 발생했습니다. 엑셀 형식과 시트를 확인해 주세요.")
                st.exception(e)
                return

    from modules.loader import get_year_to_file, load_energy_files

    st.markdown("#### 인식된 연도별 파일 목록")
    merged = get_year_to_file()
    if not merged:
        st.info("현재 인식된 에너지 사용량 파일이 없습니다.")
    else:
        rows = [
            {"연도": year, "파일명": getattr(f, "name", str(f))}
            for year, f in sorted(merged.items())
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    if (df_raw_all is None or df_raw_all.empty) and merged:
        try:
            year_to_raw_tmp, df_raw_all_tmp = load_energy_files(merged)
            st.session_state["year_to_raw_all_v4"] = year_to_raw_tmp
            st.session_state["df_raw_all_v4"] = df_raw_all_tmp
            df_raw_all = df_raw_all_tmp
            st.success(f"df_raw가 새로 생성되었습니다. 전체 행 수: {len(df_raw_all)}")
        except Exception as e:
            st.error("df_raw 생성 중 오류가 발생했습니다. 엑셀 형식을 확인해 주세요.")
            st.exception(e)
            return

    if df_raw_all is None or df_raw_all.empty:
        st.info("아직 분석 가능한 df_raw 데이터가 없습니다. 먼저 연도별 파일을 업로드해 주세요.")
        return

    try:
        from modules.analyzer import build_data1_tables
        # 업로드 탭의 기본 백데이터 분석은 소비 에너지원 기준으로 집계
        df_cons_only = df_raw_all[~df_raw_all["에너지종류"].isin(RENEWABLE_FUELS_ORDER)].copy()
        if df_cons_only.empty:
            df_cons_only = df_raw_all.copy()
        tbl_usage, tbl_area, tbl_avg3 = build_data1_tables(df_cons_only)
    except Exception as e:
        st.error("data_1(백데이터 분석) 표 생성 중 오류가 발생했습니다.")
        st.exception(e)
        return

    no_format_for_label = {"구분": ""}

    st.markdown("### 1. 연도×기관 에너지 사용량 (연단위)")
    tbl_usage_fmt = format_table(
        tbl_usage, fmt_rules, column_fmt_map=no_format_for_label, default_fmt_name="integer_comma"
    )
    st.dataframe(tbl_usage_fmt, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 2. 연도×기관 연면적")
    tbl_area_fmt = format_table(
        tbl_area, fmt_rules, column_fmt_map=no_format_for_label, default_fmt_name="integer_comma"
    )
    st.dataframe(tbl_area_fmt, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3. 연도별 3개년 평균 에너지 사용량")
    tbl_avg3_fmt = format_table(
        tbl_avg3, fmt_rules, column_fmt_map=no_format_for_label, default_fmt_name="integer_comma"
    )
    st.dataframe(tbl_avg3_fmt, use_container_width=True, hide_index=True)


# ===========================================================
# 🔧 디버그 / 진단 탭 렌더링 (이스터에그 5회 클릭 토스트 완벽 복구)
# ===========================================================
def render_debug_tab(
    year_to_raw: Mapping[int, pd.DataFrame],
    df_raw_all: Optional[pd.DataFrame],
) -> None:
    col_title, col_secret = st.columns([30, 1])

    with col_title:
        st.subheader("df_raw 메타 정보")

    with col_secret:
        if "easter_egg_count" not in st.session_state:
            st.session_state["easter_egg_count"] = 0

        if st.button("·", key="easter_egg_trigger", help=None):
            st.session_state["easter_egg_count"] += 1
            if st.session_state["easter_egg_count"] >= 5:
                st.session_state["show_easter_egg"] = True
                st.session_state["easter_egg_count"] = 0

    if st.session_state.get("show_easter_egg", False):
        st.markdown(
            """
            <style>
            @keyframes easterEggFade {
                0% { opacity: 0; transform: translate(-50%, -50%) scale(0.95); }
                10% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                85% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                100% { opacity: 0; transform: translate(-50%, -50%) scale(0.95); visibility: hidden; }
            }
            .easter-egg-toast {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 999999;
                background: rgba(35, 38, 45, 0.96);
                color: white;
                padding: 20px 32px;
                border-radius: 12px;
                font-size: 17px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.40);
                white-space: nowrap;
                animation: easterEggFade 5s ease-in-out forwards;
                pointer-events: none;
            }
            </style>
            <div class="easter-egg-toast">
                ★짜☆잔★ 프로그램 제작자: 한국보훈복지의료공단 김두호 과장 ★짜☆잔★
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["show_easter_egg"] = False

    if not year_to_raw or df_raw_all is None or df_raw_all.empty:
        st.info("아직 df_raw 데이터가 없습니다.")
        return

    years_available = sorted(year_to_raw.keys())
    st.write("로딩된 연도:", years_available)

    info_rows = [
        {"연도": year, "행 수": len(df), "기관 수": df["기관명"].nunique()}
        for year, df in year_to_raw.items()
    ]
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
# 메인 함수
# ===========================================================
def main() -> None:
    # 1. 제목: Bohun Energy Dashboard(보훈공단 에너지 관리) / 바로가기 링크 표시 삭제
    st.set_page_config(
        page_title="Bohun Energy Dashboard(보훈공단 에너지 관리)",
        layout="wide",
    )

    st.title("Bohun Energy Dashboard(보훈공단 에너지 관리)")

    # 라디오 버튼 상태 초기화 (실제 라디오 버튼 위젯은 아래 '📊 대시보드' 탭 내부 최상단에 위치)
    if "dashboard_radio_mode" not in st.session_state:
        st.session_state["dashboard_radio_mode"] = "에너지 소비량 대시보드"

    is_production = (
        st.session_state.get("dashboard_radio_mode") == "신재생에너지 생산량 대시보드"
    )

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
    year_to_raw: Dict[int, pd.DataFrame] = st.session_state.get("year_to_raw_all_v4", {})
    df_raw_all: Optional[pd.DataFrame] = st.session_state.get("df_raw_all_v4")

    year_to_file = get_year_to_file()

    cache_needs_reload = (
        not year_to_raw
        or df_raw_all is None
        or df_raw_all.empty
        or "에너지종류" not in df_raw_all.columns
    )
    if year_to_file and cache_needs_reload:
        try:
            year_to_raw, df_raw_all = load_energy_files(year_to_file)
            st.session_state["year_to_raw_all_v4"] = year_to_raw
            st.session_state["df_raw_all_v4"] = df_raw_all
        except Exception as e:
            st.warning(
                "에너지 사용량 파일을 읽는 중 오류가 발생했습니다. "
                "업로드 탭에서 파일 목록과 형식을 다시 확인해 주세요."
            )
            st.exception(e)
            year_to_raw = {}
            df_raw_all = None
    elif not year_to_file:
        year_to_raw = {}
        df_raw_all = None

    if year_to_raw:
        years_available = sorted(year_to_raw.keys())
    else:
        years_available = []

    # -------------------------------------------------------
    # 2. 사이드바 필터 (소비량/생산량 대시보드 선택에 따라 에너지 종류 상호 분리)
    # -------------------------------------------------------
    with st.sidebar:
        st.header("필터")

        view_mode = st.radio("보기 범위", ["공단 전체", "기관별"], index=0)

        if years_available:
            default_year = max(years_available)
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

        # 에너지 소비량 대시보드 선택 시 신재생에너지(태양광, 지열) 숨김,
        # 신재생에너지 생산량 대시보드 선택 시 소비 에너지원 숨김
        if is_production:
            allowed_order = list(RENEWABLE_FUELS_ORDER)
            excluded_set = set(CONSUMPTION_FUELS_ORDER) | {"기타", "용수"}
        else:
            allowed_order = list(CONSUMPTION_FUELS_ORDER)
            excluded_set = set(RENEWABLE_FUELS_ORDER) | {"용수"}

        if (
            selected_year is not None
            and df_raw_all is not None
            and not df_raw_all.empty
            and "에너지종류" in df_raw_all.columns
        ):
            energy_values = (
                df_raw_all.loc[df_raw_all["연도"] == selected_year, "에너지종류"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            valid_values = set(energy_values) - excluded_set
            available_energy_types = [e for e in allowed_order if e in valid_values]
            if not is_production:
                extras = sorted(
                    e for e in valid_values
                    if e and e not in allowed_order and e != "기타"
                )
                available_energy_types.extend(extras)
        else:
            available_energy_types = list(allowed_order)

        if not available_energy_types:
            available_energy_types = list(allowed_order)

        selected_energy = st.selectbox(
            "에너지 종류",
            ["전체"] + available_energy_types,
            index=0,
            key=f"energy_type_select_{'prod' if is_production else 'cons'}",
            help="선택한 대시보드(소비량 또는 신재생 생산량)에 해당하는 에너지원만 표시 및 필터링합니다.",
        )

    # -------------------------------------------------------
    # 3. 탭 구성 ("📊 대시보드 / 📂 에너지 사용량 파일 업로드 / 🔧 디버그 / 진단")
    # -------------------------------------------------------
    tab_dashboard, tab_upload, tab_debug = st.tabs(
        ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
    )

    # 📊 대시보드 탭: 탭 바로 아래에 라디오 버튼을 배치하여 하나만 선택하도록 구성
    with tab_dashboard:
        selected_mode = st.radio(
            "대시보드 구분 선택",
            ["에너지 소비량 대시보드", "신재생에너지 생산량 대시보드"],
            horizontal=True,
            key="dashboard_radio_mode",
            label_visibility="collapsed",
        )
        is_production = (selected_mode == "신재생에너지 생산량 대시보드")

        if year_to_raw:
            filtered_year_to_raw: Dict[int, pd.DataFrame] = {}

            for year, df in year_to_raw.items():
                sub = df.copy()

                # 모드별 에너지원 엄격 분리 (소비 대시보드에서는 태양광/지열 제외, 생산에서는 소비원 제외)
                if "에너지종류" in sub.columns:
                    if is_production:
                        sub = sub[sub["에너지종류"].isin(RENEWABLE_FUELS_ORDER)]
                    else:
                        sub = sub[~sub["에너지종류"].isin(RENEWABLE_FUELS_ORDER)]

                if view_mode == "기관별" and selected_org is not None:
                    sub = sub[sub["기관명"] == selected_org]

                if selected_energy != "전체":
                    if "에너지종류" in sub.columns:
                        sub = sub[sub["에너지종류"] == selected_energy]
                    else:
                        sub = sub.iloc[0:0]

                if not sub.empty:
                    filtered_year_to_raw[year] = sub

            analysis_year_to_raw: Mapping[int, pd.DataFrame] = filtered_year_to_raw
        else:
            analysis_year_to_raw = year_to_raw

        if selected_year is None:
            st.info(
                "아직 분석 가능한 df_raw 데이터가 없습니다. "
                "먼저 '📂 에너지 사용량 파일 업로드' 탭에서 연도별 파일을 업로드해 주세요."
            )
        else:
            render_dashboard_tab(
                spec=spec,
                fmt_rules=fmt_rules,
                analysis_year_to_raw=analysis_year_to_raw,
                selected_year=selected_year,
                view_mode=view_mode,
                selected_org=selected_org,
                is_production=is_production,
            )

    # 📂 업로드 탭
    with tab_upload:
        render_upload_tab(spec, fmt_rules, df_raw_all=df_raw_all)

    # 🔧 디버그 탭 (데이터 유무와 관계없이 5회 클릭 이스터에그 상시 작동)
    with tab_debug:
        render_debug_tab(year_to_raw, df_raw_all)


if __name__ == "__main__":
    main()
