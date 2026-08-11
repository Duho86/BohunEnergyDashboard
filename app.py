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
        import streamlit as _st  # 로컬 별칭
    except ImportError:
        return

    # 최신 버전: st.rerun 존재
    if hasattr(_st, "rerun"):
        _st.rerun()
    # 구 버전 호환: experimental_rerun 만 있는 경우
    elif hasattr(_st, "experimental_rerun"):
        _st.experimental_rerun()
    # 둘 다 없으면 아무 것도 하지 않음
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
    )
    from modules.analyzer import (
        build_data_2_usage_analysis,
        build_data_3_feedback,
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
    """data/ 폴더에서 연도 정보를 가진 엑셀 파일을 찾아 {연도: Path} 매핑 생성.

    동일 연도에 여러 파일이 있을 경우 **가장 최근에 수정된 파일**을 선택한다.
    (예: 사용자가 2024년 파일을 다시 업로드하여 교체한 경우 최신 파일 사용)
    """
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
        # mtime 기준 가장 최신 파일 선택
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
# 숫자 포맷팅 (master_energy_spec.formatting_rules 기반)
# ===========================================================
def format_number(value, rule: Mapping) -> str:
    """spec.formatting_rules 의 규칙을 적용해 숫자를 문자열로 변환."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"

    try:
        v = float(value)
    except Exception:
        return str(value)

    # ×100 옵션
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

    # NaN 제거
    s = series.dropna()
    if s.empty:
        st.info(f"{title}를(을) 표시할 데이터가 없습니다.")
        return

    # 증감률 등 음수 가능 지표는 절대값으로 비교
    if use_abs:
        s = s.abs()

    # 파이차트는 0/음수 불가 → 0 제거
    s = s[s > 0]
    if s.empty:
        st.info(f"{title}를(을) 표시할 유효한 값이 없습니다.")
        return

    # 값 큰 순으로 정렬 (높은 → 낮은)
    s = s.sort_values(ascending=False)

    # 🔴 더 이상 상위 10개 + 기타로 묶지 않음 → 전체 소속기구 그대로 사용
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
# 🔎 AI 자동 피드백 생성 유틸
# ===========================================================
def _fmt_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{digits}f}%"


def _fmt_energy(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}"


def generate_global_feedback_text(
    selected_year: int,
    df3_overall: pd.DataFrame,
    data2_overall: pd.DataFrame,
    data2_by_org: pd.DataFrame,
    df3_by_org: pd.DataFrame,
) -> str:
    """공단 전체 기준 종합분석 텍스트 생성"""

    if df3_overall is None or df3_overall.empty:
        return "데이터가 부족하여 공단 전체 종합분석을 생성할 수 없습니다."

    row_overall_fb = df3_overall.iloc[0]
    row_overall_usage = data2_overall.iloc[0]

    target = row_overall_fb.get("권장 에너지 사용량", np.nan)
    year_change = row_overall_usage.get("전년대비 증감률", np.nan)
    avg3_change = row_overall_usage.get(
        "3개년 평균 에너지 사용량 대비 증감률", np.nan
    )

    # 증가율 TOP3 (3개년 평균 대비 증감률)
    inc_list: list[str] = []
    if "3개년 평균 에너지 사용량 대비 증감률" in data2_by_org.columns:
        growth = data2_by_org[
            "3개년 평균 에너지 사용량 대비 증감률"
        ].dropna()
        growth_top = growth.sort_values(ascending=False).head(3)
        inc_list = [
            f"{org} ({_fmt_pct(val)})" for org, val in growth_top.items()
        ]

    # 면적대비 사용량 TOP3
    area_list: list[str] = []
    if "면적대비 에너지 사용비율" in data2_by_org.columns:
        upa = data2_by_org["면적대비 에너지 사용비율"].dropna()
        upa_top = upa.sort_values(ascending=False).head(3)
        area_list = [
            f"{org} ({_fmt_pct(val)})" for org, val in upa_top.items()
        ]

    # 전체 추세 판단
    if pd.isna(year_change) or pd.isna(avg3_change):
        summary = "데이터가 충분하지 않아 추세 판단이 어렵습니다."
    else:
        if year_change > 0 and avg3_change > 0:
            summary = (
                "전년 및 최근 3개년 평균 대비 에너지 사용량이 모두 증가하는 "
                "추세입니다."
            )
        elif year_change < 0 and avg3_change < 0:
            summary = (
                "전년 및 최근 3개년 평균 대비 에너지 사용량이 모두 감소하는 "
                "추세입니다."
            )
        elif year_change > 0 and avg3_change <= 0:
            summary = (
                "전년 대비로는 소폭 증가했지만, 최근 3개년 평균 기준으로는 "
                "안정 또는 감소 추세입니다."
            )
        elif year_change < 0 and avg3_change >= 0:
            summary = (
                "전년 대비로는 감소했지만, 최근 3개년 평균 기준으로는 "
                "여전히 높은 수준을 유지하고 있습니다."
            )
        else:
            summary = (
                "전년 대비와 최근 3개년 평균 대비 추세가 상이하여 "
                "세부 원인 분석이 필요합니다."
            )

    # 이슈 기관: 관리대상(O) 중 사용 분포 순위가 높은 기관
    issue_org = None
    tmp = df3_by_org.copy()
    if "에너지 사용량 관리 대상" in tmp.columns:
        tmp = tmp[tmp["에너지 사용량 관리 대상"] == "O"]
    if "사용 분포 순위" in tmp.columns and not tmp.empty:
        tmp = tmp.sort_values("사용 분포 순위")  # 1위가 가장 높은 비중
        if not tmp.empty:
            issue_org = tmp.index[0]
    issue_org_text = issue_org if issue_org else "특정 기관"

    lines: list[str] = []
    lines.append(
        f"{selected_year}년 권장 에너지 사용량: "
        f"**{_fmt_energy(target)} kWh**"
    )
    lines.append(f"전년 대비 증감률: **{_fmt_pct(year_change)}**")
    lines.append(
        f"최근 3개년 평균 대비 증감률: **{_fmt_pct(avg3_change)}**\n"
    )

    lines.append("**● 관리대상 기관 자동 탐지**")
    lines.append(
        "- 증가율이 높은 기관: "
        + (", ".join(inc_list) if inc_list else "해당 없음")
    )
    lines.append(
        "- 면적 대비 사용량이 높은 기관: "
        + (", ".join(area_list) if area_list else "해당 없음")
    )
    lines.append("")
    lines.append("**● 종합판단(자동 문구)**")
    lines.append(
        f"공단 전체적으로는 {summary} "
        f"특히 **{issue_org_text}**의 에너지 사용 수준에 대한 "
        "면밀한 모니터링이 필요합니다."
    )

    return "\n".join(lines)


def generate_institution_feedback_text(
    org_name: str,
    row2: pd.Series,
    row3: pd.Series,
    upa_mean: float,
    total_orgs: int,
) -> str:
    """소속기구별 맞춤형 피드백 텍스트 생성"""

    upa = row2.get("면적대비 에너지 사용비율", np.nan)
    vs3 = row2.get("3개년 평균 에너지 사용량 대비 증감률", np.nan)
    rank_share = row3.get("사용 분포 순위", np.nan)

    # 증가/감소 추세
    if pd.isna(vs3) or abs(vs3) < 0.001:
        trend_word = "유지"
    elif vs3 > 0:
        trend_word = "증가"
    else:
        trend_word = "감소"

    # 공단 평균 대비 수준
    if pd.isna(upa) or pd.isna(upa_mean):
        level_word = "평가 불가"
    elif upa > upa_mean * 1.05:
        level_word = "공단 평균 대비 **높은** 수준"
    elif upa < upa_mean * 0.95:
        level_word = "공단 평균 대비 **낮은** 수준"
    else:
        level_word = "공단 평균과 **유사한** 수준"

    # 비중 순위
    if pd.isna(rank_share):
        rank_text = "순위 정보 없음"
    else:
        rank_text = f"{int(rank_share)}/{total_orgs}"

    # 조건별 제안 문구
    suggestions: list[str] = []
    if not pd.isna(vs3) and vs3 > 0:
        suggestions.append(
            "• 증가 요인(증축, 운영시간 증가 등)을 분석하고 "
            "절감 목표를 재설정할 필요가 있습니다."
        )
    if not pd.isna(upa) and not pd.isna(upa_mean) and upa > upa_mean * 1.05:
        suggestions.append(
            "• 냉난방 효율, 단열 상태, 운영 기준 등을 점검하여 "
            "연면적 대비 에너지 효율을 개선해야 합니다."
        )
    if not pd.isna(rank_share) and rank_share <= 5:
        suggestions.append(
            "• 공단 전체 목표 달성에 미치는 영향이 큰 기관으로, "
            "피크타임 절감 및 자동제어 강화가 요구됩니다."
        )
    if not suggestions:
        suggestions.append(
            "• 현재 수준을 유지하면서 에너지 절감 잠재 영역을 "
            "지속적으로 발굴하는 것이 필요합니다."
        )

    lines: list[str] = []
    lines.append(f"#### ▶ {org_name}")
    lines.append("**1) 에너지 사용 요약**")
    lines.append(f"- 연면적 대비 사용량: {_fmt_pct(upa)}")
    lines.append(f"- 3개년 평균 대비 증감률: {_fmt_pct(vs3)}")
    lines.append(f"- 에너지 사용 비중 순위: {rank_text}")
    lines.append("")
    lines.append("자동 문구:")
    lines.append(
        "> 최근 3개년 평균 대비 에너지 사용량이 "
        f"**{trend_word}** 추세를 보이고 있으며, "
        f"연면적 대비 사용량은 {level_word}입니다."
    )
    lines.append("")
    lines.append("**2) 기관 맞춤형 제안**")
    lines.extend(suggestions)

    return "\n".join(lines)


def generate_common_recommendations_text(
    df3_by_org: pd.DataFrame,
    data2_by_org: pd.DataFrame,
) -> str:
    """공단 공통 제안 텍스트 생성"""

    targets: list[str] = []
    if "에너지 사용량 관리 대상" in df3_by_org.columns:
        targets = list(
            df3_by_org[df3_by_org["에너지 사용량 관리 대상"] == "O"].index
        )

    lines: list[str] = []
    lines.append(
        "다음 제안은 공단 전체 기관에 공통으로 적용할 수 있는 "
        "에너지 절감 방향입니다.\n"
    )
    lines.append(
        "- 설비 노후가 의심되는 기관(관리대상 및 면적대비 사용량 상위 기관)을 "
        "**우선 대상으로** 고효율 설비 교체 로드맵을 수립합니다."
    )
    lines.append(
        "- 보훈병원 및 보훈요양원 등 상시 운영시설에는 "
        "**BEMS(건물 에너지 관리 시스템)** 적용 및 데이터 기반 모니터링을 확대합니다."
    )
    lines.append(
        "- 전 기관을 대상으로 **대기전력 절감 캠페인, 불필요 조명 소등, "
        "설정온도 표준화** 등을 정착시킵니다."
    )
    if targets:
        lines.append(
            f"- 에너지 사용량 관리 대상 기관({', '.join(targets)})은 "
            "월별 사용량을 집중 모니터링하고, 현장 점검과 절감 컨설팅을 "
            "우선 지원합니다."
        )
    else:
        lines.append(
            "- 현재 관리대상으로 분류된 기관은 없으나, 사용량 추세를 주기적으로 "
            "점검하여 이상징후를 조기에 발견할 필요가 있습니다."
        )

    return "\n".join(lines)


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
    # 0. 에너지 사용량 추이 (그래프 섹션)
    # -------------------------------------------------------
    st.subheader("에너지 사용량 추이")

    try:
        df_list = [
            df.copy()
            for df in analysis_year_to_raw.values()
            if df is not None and not df.empty
        ]
        if df_list:
            df_all = pd.concat(df_list, ignore_index=True)
        else:
            df_all = pd.DataFrame()
    except Exception as e:
        st.warning("그래프용 df_raw 병합 중 오류가 발생했습니다.")
        st.exception(e)
        df_all = pd.DataFrame()

    col_g1, col_g2 = st.columns(2)

    # 월별 에너지 사용량 추이 (라인 그래프)
    with col_g1:
        st.markdown("**월별 에너지 사용량 추이**")
        if df_all.empty or "연도" not in df_all.columns:
            st.info("월별 그래프를 그릴 df_raw 데이터가 없습니다.")
        else:
            df_year = df_all[df_all["연도"] == selected_year].copy()

            # 예: "1월", "1월 사용량", "1 월" 등 모두 인식
            month_info = []
            for c in df_year.columns:
                m = re.search(r"(\d{1,2})\s*월", str(c))
                if m:
                    month_num = int(m.group(1))
                    if 1 <= month_num <= 12:
                        month_info.append((month_num, c))

            if not month_info:
                st.info(
                    "1월~12월 관련 컬럼을 찾지 못해 월별 에너지 사용량 그래프를 표시할 수 없습니다."
                )
            else:
                month_info.sort(key=lambda x: x[0])
                month_nums = [m for m, _ in month_info]
                month_cols = [c for _, c in month_info]

                for c in month_cols:
                    df_year[c] = pd.to_numeric(df_year[c], errors="coerce")

                monthly = df_year[month_cols].sum(axis=0)
                monthly.index = month_nums  # 1~12 숫자 인덱스
                st.line_chart(monthly)

    # 연도별 에너지 사용량 추이 (막대 그래프, 최대 5개년)
    with col_g2:
        st.markdown("**연도별 에너지 사용량 추이 (최대 5개년)**")
        if df_all.empty or "연도" not in df_all.columns:
            st.info("연도별 그래프를 그릴 df_raw 데이터가 없습니다.")
        else:
            if "연단위" not in df_all.columns:
                st.info("연단위 컬럼이 없어 연도별 에너지 사용량을 계산할 수 없습니다.")
            else:
                yearly = (
                    df_all.groupby("연도", dropna=False)["연단위"]
                    .sum()
                    .sort_index()
                )
                yearly = yearly.tail(5)
                if yearly.empty:
                    st.info("연도별 에너지 사용량 합계를 계산할 수 없습니다.")
                else:
                    st.bar_chart(yearly)

    # 반기별 / 분기별 에너지 사용량 추이
    col_g3, col_g4 = st.columns(2)

    def _selected_year_monthly_totals() -> pd.Series:
        if df_all.empty or "연도" not in df_all.columns:
            return pd.Series(dtype=float)

        df_selected = df_all[df_all["연도"] == selected_year].copy()
        if df_selected.empty:
            return pd.Series(dtype=float)

        month_map: Dict[int, str] = {}
        for c in df_selected.columns:
            m = re.search(r"(\d{1,2})\s*월", str(c))
            if m:
                month_num = int(m.group(1))
                if 1 <= month_num <= 12 and month_num not in month_map:
                    month_map[month_num] = c

        if len(month_map) < 12:
            return pd.Series(dtype=float)

        values = {}
        for month_num in range(1, 13):
            col = month_map[month_num]
            values[month_num] = pd.to_numeric(
                df_selected[col], errors="coerce"
            ).fillna(0).sum()
        return pd.Series(values, dtype=float)

    monthly_totals = _selected_year_monthly_totals()

    with col_g3:
        st.markdown("**반기별 에너지 사용량 추이**")
        if monthly_totals.empty:
            st.info("반기별 에너지 사용량을 계산할 월별 데이터가 없습니다.")
        else:
            half_yearly = pd.Series(
                {
                    "상반기": monthly_totals.loc[1:6].sum(),
                    "하반기": monthly_totals.loc[7:12].sum(),
                }
            )
            st.bar_chart(half_yearly)

    with col_g4:
        st.markdown("**분기별 에너지 사용량 추이**")
        if monthly_totals.empty:
            st.info("분기별 에너지 사용량을 계산할 월별 데이터가 없습니다.")
        else:
            quarterly = pd.Series(
                {
                    "1분기": monthly_totals.loc[1:3].sum(),
                    "2분기": monthly_totals.loc[4:6].sum(),
                    "3분기": monthly_totals.loc[7:9].sum(),
                    "4분기": monthly_totals.loc[10:12].sum(),
                }
            )
            st.bar_chart(quarterly)

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
        st.error("에너지 사용량 분석(data_2) 계산 중 오류가 발생했습니다.")
        st.exception(e)
        return

    data2_overall = data2.overall.copy()
    data2_by_org = data2.by_org.copy()

    org_order = list(get_org_order())

    # 보기 범위에 따른 기관 정렬 / 필터
    if view_mode == "공단 전체":
        data2_by_org = data2_by_org.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        if selected_org in data2_by_org.index:
            data2_by_org = data2_by_org.loc[[selected_org]]
        else:
            data2_by_org = data2_by_org.iloc[0:0]

    # 포맷 규칙 매핑
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

    # 3) 공단 전체 기준 표 포맷 (시설구분 포함)
    df2_overall_fmt = format_table(
        data2_overall,
        fmt_rules,
        DATA2_OVERALL_FMT,
    )
    # 4) 공단 전체 기준 표에서는 시설구분 3개 컬럼 제거
    for col in fac_cols:
        if col in df2_overall_fmt.columns:
            df2_overall_fmt = df2_overall_fmt.drop(columns=[col])

    # 5) 소속기구별 표 포맷
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
        if fac_overall_fmt is not None and not fac_overall_fmt.empty:
            st.dataframe(fac_overall_fmt, use_container_width=True)
        else:
            st.info("시설구분별 데이터가 없습니다.")

    st.markdown("---")

    # -------------------------------------------------------
    # 1-1. 소속기구별 분석 원그래프 (에너지 분석 부문)
    # -------------------------------------------------------
    st.markdown("**소속기구별 원그래프 (에너지 분석 부문)**")

    if data2_by_org is None or data2_by_org.empty or len(data2_by_org.index) < 2:
        st.info("소속기구별 비교를 위한 데이터가 2개 미만입니다.")
    else:
        pie_metrics = [
            ("에너지 사용량", "에너지 사용량", False),
            ("면적대비 에너지 사용비율", "면적대비 에너지 사용비율", False),
            ("에너지 사용 비중", "에너지 사용 비중", False),
            ("3개년 평균 에너지 사용량 대비 증감률", "3개년 평균 에너지 사용량 대비 증감률", True),
            ("시설별 평균 면적 대비 에너지 사용비율", "시설별 평균 면적 대비 에너지 사용비율", False),
        ]

        # 2개씩 좌우 분할
        for i in range(0, len(pie_metrics), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(pie_metrics):
                    break
                title_kor, col_name, use_abs = pie_metrics[i + j]
                with cols[j]:
                    if col_name in data2_by_org.columns:
                        series = data2_by_org[col_name]
                        render_pie_from_series(series, title_kor, use_abs=use_abs)
                    else:
                        st.info(f"'{col_name}' 컬럼이 없어 원그래프를 표시할 수 없습니다.")

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
        st.error("피드백(data_3) 계산 중 오류가 발생했습니다.")
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

    # 2-1. 표 포맷팅 및 기관별 필터
    df3_overall_fmt = format_table(
        data3.overall,
        fmt_rules,
        DATA3_OVERALL_FMT,
    )

    df3_by_org = data3.by_org.copy()
    df3_detail = data3.detail.copy()

    org_order = list(get_org_order())

    if view_mode == "공단 전체":
        df3_by_org = df3_by_org.reindex(org_order)
        df3_detail = df3_detail.reindex(org_order)
    elif view_mode == "기관별" and selected_org:
        if selected_org in df3_by_org.index:
            df3_by_org = df3_by_org.loc[[selected_org]]
        else:
            df3_by_org = df3_by_org.iloc[0:0]
        if selected_org in df3_detail.index:
            df3_detail = df3_detail.loc[[selected_org]]
        else:
            df3_detail = df3_detail.iloc[0:0]

    df3_by_org_fmt = format_table(
        df3_by_org,
        fmt_rules,
        DATA3_BYORG_FMT,
    )

    st.markdown("**1. 공단 전체 기준**")
    st.dataframe(df3_overall_fmt, use_container_width=True)
    st.caption("* 온실가스감축목표(NDC) 연평균 감축률 4.17% 기준")


    st.markdown("---")
    st.markdown("**2. 소속기구별**")

    # -------------------------------------------------------
    # 2-1. 사용 분포 순위 원그래프 (에너지 3개년 평균 증가 순위 / 평균 에너지 사용량(연면적 기준) 순위)
    # -------------------------------------------------------
    if df3_by_org is None or df3_by_org.empty or len(df3_by_org.index) < 2:
        st.info("순위 비교를 위한 데이터가 2개 미만입니다.")
    else:
        st.markdown("**소속기구별 원그래프 (사용 분포 순위)**")

        rank_metrics = [
            ("에너지 3개년 평균 증가 순위", "에너지 3개년 평균 증가 순위"),
            ("평균 에너지 사용량(연면적 기준) 순위", "평균 에너지 사용량(연면적 기준) 순위"),
        ]

        cols = st.columns(2)
        for idx, (title_kor, col_name) in enumerate(rank_metrics):
            with cols[idx]:
                if col_name in df3_by_org.columns:
                    rank_series = df3_by_org[col_name].dropna()
                    if rank_series.empty:
                        st.info(f"'{col_name}' 데이터가 없습니다.")
                    else:
                        # 순위는 숫자가 작을수록 상위이므로,
                        # (최대+1-순위)로 점수를 만들어 파이 비중에 사용
                        max_rank = rank_series.max()
                        score = (max_rank + 1) - rank_series
                        render_pie_from_series(score, title_kor, use_abs=False)
                else:
                    st.info(f"'{col_name}' 컬럼이 없어 원그래프를 표시할 수 없습니다.")

    st.dataframe(df3_by_org_fmt, use_container_width=True)

    st.markdown("---")
    st.markdown("**3. 에너지 사용량 관리 대상 상세**")

    if df3_detail is None or df3_detail.empty:
        st.info("관리 대상 상세 데이터를 생성할 수 없습니다. (데이터 부족 또는 분석 오류)")
    else:
        st.dataframe(df3_detail, use_container_width=True)

    # -------------------------------------------------------
    # 3. AI 제안 피드백 (맨 아래 배치)
    # -------------------------------------------------------
    #st.markdown("---")
    #st.subheader("AI 제안 피드백")

    # (1) 종합분석 텍스트 생성 (기존 서술형 내용)
    try:
        overall_row = data3.overall.iloc[0]
        rec_usage = float(overall_row.get("권장 에너지 사용량", np.nan))
        red_yoy = float(overall_row.get("전년대비 감축률", np.nan))
        red_vs3 = float(overall_row.get("3개년 대비 감축률", np.nan))

        df_detail_tmp = data3.detail.copy()
        risk_mask = (df_detail_tmp == "O").any(axis=1)
        risk_orgs = df_detail_tmp.index[risk_mask].tolist()

        comment_parts: list[str] = []
        if not np.isnan(rec_usage):
            comment_parts.append(
                f"{selected_year}년 권장 에너지 사용량은 약 {rec_usage:,.0f}kWh 입니다."
            )
        if not np.isnan(red_yoy):
            comment_parts.append(
                f"전년 대비 감축 목표는 {red_yoy * 100:.1f}% 수준입니다."
            )
        if not np.isnan(red_vs3):
            comment_parts.append(
                f"최근 3개년 평균 대비 감축 목표는 {red_vs3 * 100:.1f}% 수준입니다."
            )
        if risk_orgs:
            comment_parts.append("관리대상으로 분류된 기관: " + ", ".join(risk_orgs))

        if comment_parts:
            summary_text = "\n".join(f"* {t}" for t in comment_parts)
        else:
            summary_text = "* 피드백을 생성할 수 있는 데이터가 부족합니다."
    except Exception:
        summary_text = "* 종합분석 정보를 불러오는 중 오류가 발생했습니다."

    # (2) 에너지 절감을 위한 제안 (고정 텍스트 – GPT 판단 기반 템플릿)
    #st.markdown("**1. 공단 전체 기준**")
    #st.dataframe(df3_overall_fmt, use_container_width=True)

    #st.markdown("---")
    #st.markdown("**2. 소속기구별**")
    #st.dataframe(df3_by_org_fmt, use_container_width=True)

    #st.markdown("---")
    #st.markdown("**3. 에너지 사용량 관리 대상 상세**")
    #df3_detail = data3.detail.copy().reindex(org_order)
    #st.dataframe(df3_detail, use_container_width=True)

    # ---------------------------------------------------
    # 4. AI 제안 피드백 (자동 생성 텍스트)
    # ---------------------------------------------------
    st.markdown("---")
    st.subheader("AI 제안 피드백")

    # (종합분석)
    st.markdown("#### (종합분석)")
    global_text = generate_global_feedback_text(
        selected_year=selected_year,
        df3_overall=data3.overall,
        data2_overall=data2_overall,
        data2_by_org=data2_by_org,
        df3_by_org=df3_by_org,
    )
    st.markdown(global_text)

    # 소속기구별 맞춤형 피드백
    st.markdown("---")
    st.markdown("#### [소속기구별 맞춤형 피드백]")

    upa_mean = data2_by_org["면적대비 에너지 사용비율"].mean()
    total_orgs = len(data2_by_org)

    for org in data2_by_org.index:
        row2 = data2_by_org.loc[org]
        row3 = df3_by_org.loc[org]
        inst_text = generate_institution_feedback_text(
            org_name=org,
            row2=row2,
            row3=row3,
            upa_mean=upa_mean,
            total_orgs=total_orgs,
        )
        st.markdown(inst_text)
        st.markdown("")

    # 공단 공통 제안
    st.markdown("---")
    st.markdown("#### [에너지 절감을 위한 공단 공통 제안]")
    st.markdown(
        generate_common_recommendations_text(
            df3_by_org=df3_by_org,
            data2_by_org=data2_by_org,
        )
    )



# ===========================================================
# 📂 업로드 탭 렌더링
# ===========================================================
def _compute_upload_signature(uploaded_files) -> Optional[str]:
    """
    업로드된 파일 목록에 대한 '변경 여부' 식별용 시그니처.

    - 파일명 + 파일 크기 + 내용(md5)을 합쳐 해시
    - 같은 파일 세트면 시그니처도 항상 동일
    """
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
        "df_raw(U/V/W 기반이 아닌 연단위 기준)로 변환하여 분석에 사용합니다."
    )

    # 1) 파일 업로드 위젯
    uploaded_files = st.file_uploader(
        "연도별 에너지 사용량 파일 업로드 (여러 개 선택 가능)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    # 현재 업로드 세트의 시그니처 계산
    current_sig = _compute_upload_signature(uploaded_files) if uploaded_files else None
    last_sig = st.session_state.get("last_upload_sig")

    # 2) 새 업로드 세트일 때만 처리 (== 1회 처리 보장)
    if uploaded_files and current_sig and current_sig != last_sig:
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 업로드 이전에 존재하던 로컬 파일 목록(연도 기준)
        from modules.loader import discover_local_energy_files, get_year_to_file, load_energy_files

        local_before = discover_local_energy_files()
        before_years = set(local_before.keys())

        new_years: list[int] = []
        updated_years: list[int] = []

        # 2-1) 파일 저장 (연도별로 항상 덮어쓰기)
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

        # 세션 업로드 캐시는 더 이상 사용하지 않으므로 비움
        st.session_state["year_to_file"] = {}

        # 2-2) 저장된 파일 기준으로 year_to_raw / df_raw_all 캐시 재생성
        merged = get_year_to_file()
        if merged:
            try:
                year_to_raw_tmp, df_raw_all_tmp = load_energy_files(merged)

                st.session_state["year_to_raw_cache"] = year_to_raw_tmp
                st.session_state["df_raw_all_cache"] = df_raw_all_tmp

                df_raw_all = df_raw_all_tmp

                if new_years:
                    years_str = ", ".join(str(y) for y in sorted(set(new_years)))
                    st.success(f"새 연도 파일 등록 완료: {years_str}년")
                if updated_years:
                    years_str = ", ".join(str(y) for y in sorted(set(updated_years)))
                    st.success(f"기존 연도 파일 업데이트 완료: {years_str}년")

                # 이 업로드 세트는 처리 완료 → 같은 세트는 다시 처리하지 않도록 기록
                st.session_state["last_upload_sig"] = current_sig

            except Exception as e:
                st.error(
                    "에너지 사용량 파일 로딩 중 오류가 발생했습니다. "
                    "엑셀 형식과 시트를 확인해 주세요."
                )
                st.exception(e)
                return

    # 이미 처리된 업로드 세트라면(= current_sig == last_sig) 추가 연산 생략
    # → Streamlit 기본 rerun 1회만 일어나고, load_energy_files 재호출 없음

    # 3) 현재 인식된 파일 목록 표시
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

    # 4) df_raw_all 이 아직 없으면 (최초 실행) 로컬 파일 기준으로 한 번만 로딩
    if (df_raw_all is None or df_raw_all.empty) and merged:
        try:
            year_to_raw_tmp, df_raw_all_tmp = load_energy_files(merged)

            st.session_state["year_to_raw_cache"] = year_to_raw_tmp
            st.session_state["df_raw_all_cache"] = df_raw_all_tmp

            df_raw_all = df_raw_all_tmp

            st.success(f"df_raw가 새로 생성되었습니다. 전체 행 수: {len(df_raw_all)}")
        except Exception as e:
            st.error("df_raw 생성 중 오류가 발생했습니다. 엑셀 형식을 확인해 주세요.")
            st.exception(e)
            return

    # 5) 여전히 df_raw_all 이 없으면 표 생성 불가
    if df_raw_all is None or df_raw_all.empty:
        st.info("아직 분석 가능한 df_raw 데이터가 없습니다. 먼저 연도별 파일을 업로드해 주세요.")
        return

    # 6) data_1용 백데이터 분석 표 생성
    try:
        from modules.analyzer import build_data1_tables

        tbl_usage, tbl_area, tbl_avg3 = build_data1_tables(df_raw_all)
    except Exception as e:
        st.error("data_1(백데이터 분석) 표 생성 중 오류가 발생했습니다.")
        st.exception(e)
        return

    no_format_for_label = {"구분": ""}

    st.markdown("### 1. 연도×기관 에너지 사용량 (연단위)")
    tbl_usage_fmt = format_table(
        tbl_usage,
        fmt_rules,
        column_fmt_map=no_format_for_label,
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
# 메인 함수
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

    # 🔹 파일은 있는데 캐시가 없거나, 구버전 캐시(에너지종류 컬럼 없음)이면 강제 재로딩
    cache_needs_reload = (
        not year_to_raw
        or df_raw_all is None
        or df_raw_all.empty
        or "에너지종류" not in df_raw_all.columns
    )
    if year_to_file and cache_needs_reload:
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
            # 항상 실제 로딩된 데이터 중 가장 최근 이행연도를 기본값으로 사용한다.
            # master_energy_spec.json의 current_year 값이 오래되어도 최신 업로드 연도가 우선한다.
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

        # 업로드 데이터의 연료/에너지원 구분을 실제 필터 옵션으로 사용한다.
        energy_order = ["전기", "가스(LNG)", "등유", "지역난방"]
        if selected_year is not None and df_raw_all is not None and not df_raw_all.empty and "에너지종류" in df_raw_all.columns:
            energy_values = (
                df_raw_all.loc[df_raw_all["연도"] == selected_year, "에너지종류"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            available_energy_types = [e for e in energy_order if e in set(energy_values)]
            extras = sorted(
                e for e in set(energy_values)
                if e and e not in energy_order and e != "기타"
            )
            available_energy_types.extend(extras)
            if "기타" in set(energy_values):
                available_energy_types.append("기타")
        else:
            available_energy_types = []

        selected_energy = st.selectbox(
            "에너지 종류",
            ["전체"] + available_energy_types,
            index=0,
            help="업로드 파일의 연료/에너지원 구분에 따라 모든 대시보드 분석을 필터링합니다.",
        )

    # -------------------------------------------------------
    # 3. 탭 구성
    # -------------------------------------------------------
    tab_dashboard, tab_upload, tab_debug = st.tabs(
        ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그 / 진단"]
    )

    # 분석에 사용할 year_to_raw
    # 기관/에너지 종류 필터를 모든 연도에 동일하게 적용하여 월별·연도별·분기/반기 및 분석표가 일관되게 동작하도록 한다.
    if year_to_raw:
        filtered_year_to_raw: Dict[int, pd.DataFrame] = {}
        for year, df in year_to_raw.items():
            sub = df.copy()

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
