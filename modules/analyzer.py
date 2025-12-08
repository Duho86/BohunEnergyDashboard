from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

from .loader import load_spec, get_org_order


# ===============================
# 로그 유틸
# ===============================
def _log_error(msg: str) -> None:
    if st is not None:
        st.error(msg)
    else:
        print("[ERROR]", msg)


def _log_warning(msg: str) -> None:
    if st is not None:
        st.warning(msg)
    else:
        print("[WARN]", msg)


# ===============================
# df_raw 결합
# ===============================
def _concat_raw(year_to_raw: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """
    여러 연도의 df_raw 를 하나로 합친다.
    - 필수 컬럼: 기관명, 시설구분, 연면적, 연단위, 연도
    - '합계' 행(기관명='합계' & 연면적/연단위 NaN)은 미리 제거
    - 연도/연면적/연단위에 NaN 이 있는 행은 제외하고 경고를 출력한다.
    """
    if not year_to_raw:
        raise ValueError("year_to_raw 가 비어 있습니다. 먼저 파일을 업로드해 주세요.")

    required_cols = ["기관명", "시설구분", "연면적", "연단위", "연도"]
    dfs: List[pd.DataFrame] = []

    for year, df in year_to_raw.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()

        # 엑셀 합계행 제거
        if {"기관명", "연면적", "연단위"}.issubset(tmp.columns):
            mask_total = (
                tmp["기관명"].astype(str).str.strip().eq("합계")
                & tmp["연면적"].isna()
                & tmp["연단위"].isna()
            )
            if mask_total.any():
                tmp = tmp.loc[~mask_total].copy()

        # 필수 컬럼 체크
        for col in required_cols:
            if col not in tmp.columns:
                raise ValueError(
                    f"{year}년 df_raw에 '{col}' 컬럼이 없습니다. loader 스키마를 확인해 주세요."
                )

        # 숫자 컬럼 정제
        num_cols = ["연도", "연면적", "연단위"]
        for col in num_cols:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        # NaN 있는 행 제외 + 경고
        na_mask = tmp[num_cols].isna().any(axis=1)
        na_cnt = int(na_mask.sum())
        if na_cnt > 0:
            bad_rows = tmp.loc[na_mask, ["기관명", "연도", "연면적", "연단위"]]
            _log_warning(
                f"{year}년 df_raw에서 연도/연면적/연단위에 NaN 이 있는 행 {na_cnt}개를 분석에서 제외합니다.\n"
                f"{bad_rows.to_string(index=False)}"
            )
            tmp = tmp.loc[~na_mask].copy()

        dfs.append(tmp)

    if not dfs:
        raise ValueError("유효한 df_raw 데이터가 없습니다.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["연도"] = df_all["연도"].astype(int)
    df_all["기관명"] = df_all["기관명"].astype(str)
    df_all["시설구분"] = df_all["시설구분"].astype(str)
    return df_all


# ===============================
# data_2 – 에너지 사용량 분석
# ===============================
@dataclass
class Data2Result:
    overall: pd.DataFrame          # 공단 전체 기준
    by_org: pd.DataFrame           # 소속기구별 분석
    baseline_by_org: pd.Series     # 각 기관별 3개년 평균 (엑셀 B7 / U7 에 해당)


def _compute_overall_usage(
    df_all: pd.DataFrame, spec: dict, current_year: int
) -> pd.Series:
    """
    공단 전체 에너지 사용량 / 전년대비 / 3개년 평균 대비.

    - 기본적으로 spec["meta"]["analysis_years"] 를 사용하되,
      실제 데이터에 존재하는 추가 연도(예: 2025)가 있으면 함께 포함한다.
    - 데이터가 없는 연도는 0으로 처리한다.
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]

    # df_all 에 실제로 존재하는 연도
    years_in_data = sorted(int(y) for y in df_all["연도"].dropna().unique())
    if not years_in_data:
        raise ValueError("df_raw 에 연도 데이터가 없습니다.")

    # spec 정의 + 실제 데이터 연도 모두 포함
    years_all = sorted(set(analysis_years) | set(years_in_data))

    if current_year not in years_all:
        raise ValueError(f"{current_year}년은 분석 가능한 연도 목록에 없습니다.")

    # 연도별 연단위 합계 (없는 연도는 0으로)
    total_by_year = df_all.groupby("연도", dropna=False)["연단위"].sum()
    total_by_year = total_by_year.reindex(years_all, fill_value=0.0)

    cur = float(total_by_year.loc[current_year])

    # 직전 연도(또는 가장 가까운 과거 연도)
    prev_candidates = [y for y in years_all if y < current_year]
    if prev_candidates:
        prev_year = prev_candidates[-1]
        prev = float(total_by_year.loc[prev_year])
    else:
        prev = 0.0

    # 전년대비 증감률 (전년이 없거나 0이면 0으로)
    yoy_rate = (cur - prev) / prev if prev != 0 else 0.0

    # 직전 최대 3개년 평균 (과거 연도가 없으면 0으로)
    past_years = [y for y in years_all if y < current_year]
    three_years = past_years[-3:]
    if three_years:
        avg3 = float(total_by_year.loc[three_years].mean())
        vs3_rate = (cur - avg3) / avg3 if avg3 != 0 else 0.0
    else:
        vs3_rate = 0.0

    return pd.Series(
        {
            "에너지 사용량(현재 기준)": cur,
            "전년대비 증감률": yoy_rate,
            "3개년 평균 에너지 사용량 대비 증감률": vs3_rate,
        }
    )


def _compute_overall_by_facility(df_all: pd.DataFrame, current_year: int) -> pd.Series:
    """
    시설구분별 '면적대비 에너지 사용비율' 평균을 계산한다.

    엑셀 기준:
      - 기관별 면적대비 비율  =  연면적 / 에너지 사용량
      - 시설구분별 평균값     =  해당 시설구분 기관들의 위 비율의 산술평균
        (AVERAGEIFS(소속기구별!E열, 소속기구별!B열, 시설구분)에 해당)
    """
    df_year = df_all[df_all["연도"] == current_year].copy()
    if df_year.empty:
        raise ValueError(f"{current_year}년 데이터가 없습니다.")

    # 기관·시설구분 단위로 집계 (연단위 합, 연면적 최대값)
    grouped = (
        df_year.groupby(["기관명", "시설구분"], dropna=False)[["연단위", "연면적"]]
        .agg({"연단위": "sum", "연면적": "max"})
    )

    # 기관별 면적대비 에너지 사용비율 = 연면적 / 에너지 사용량
    area_per_usage = grouped["연면적"] / grouped["연단위"].replace(0, np.nan)
    grouped["면적대비 에너지 사용비율"] = area_per_usage

    # 시설구분별 평균 (단순 평균)
    fac_mean = (
        grouped.groupby("시설구분", dropna=False)["면적대비 에너지 사용비율"]
        .mean()
    )

    return pd.Series(
        {
            "의료시설": float(fac_mean.get("의료시설", np.nan)),
            "복지시설": float(fac_mean.get("복지시설", np.nan)),
            "기타시설": float(fac_mean.get("기타시설", np.nan)),
        }
    )


def _compute_org_level_current_metrics(
    df_all: pd.DataFrame,
    spec: dict,
    current_year: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    2. 에너지 사용량 분석 – 소속기구별 표 전체를 계산한다.

    반환:
      df_org       : 각 기관별 지표 DataFrame
      baseline_cur : 각 기관별 3개년 평균 에너지 사용량 (현재연도 기준)
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]

    # 실제 데이터에 존재하는 연도와 합집합
    years_in_data = sorted(int(y) for y in df_all["연도"].dropna().unique())
    years_all = sorted(set(analysis_years) | set(years_in_data))

    # 피벗에 사용할 연도는 실제 데이터가 있는 연도들
    years = [y for y in years_all if y in df_all["연도"].unique()]

    if current_year not in years:
        raise ValueError(f"{current_year}년 데이터가 df_all 에 없습니다.")

    # === 연도·기관별 사용량 피벗 (3개년 평균 계산용) ===
    usage_by_year_org = (
        df_all.groupby(["기관명", "연도"], dropna=False)["연단위"]
        .sum()
        .unstack("연도", fill_value=0.0)
        .reindex(columns=years, fill_value=0.0)
    )

    # === 현재연도 기준 기관별 메타 정보 (연면적, 시설구분) ===
    df_cur = df_all[df_all["연도"] == current_year].copy()
    if df_cur.empty:
        raise ValueError(f"{current_year}년 데이터가 없습니다.")

    area_by_org = df_cur.groupby("기관명", dropna=False)["연면적"].max()
    fac_type_by_org = df_cur.groupby("기관명", dropna=False)["시설구분"].first()

    # 기관 목록 정렬 (spec 의 org_order 사용)
    org_order = list(get_org_order())
    usage_by_year_org = usage_by_year_org.reindex(org_order, fill_value=0.0)
    area_by_org = area_by_org.reindex(org_order).fillna(0.0)
    fac_type_by_org = fac_type_by_org.reindex(org_order)

    # === 3개년 평균 (연도별 각 소속기구별 평균) ===
    baseline_by_year_org = pd.DataFrame(
        index=usage_by_year_org.index, columns=years, dtype=float
    )
    for i, y in enumerate(years):
        # 해당 연도보다 이전의 최대 3개년
        prev_years = years[:i][-3:]
        if prev_years:
            baseline_by_year_org[y] = usage_by_year_org[prev_years].mean(axis=1)
        else:
            baseline_by_year_org[y] = usage_by_year_org[y]

    if current_year not in baseline_by_year_org.columns:
        raise ValueError(f"{current_year}년의 3개년 평균 기준을 계산할 수 없습니다.")

    usage_cur = usage_by_year_org[current_year]
    baseline_cur = baseline_by_year_org[current_year]

    # 3개년 평균 에너지 사용량 대비 증감률 (기준이 0이면 0으로)
    vs3 = (usage_cur - baseline_cur) / baseline_cur.replace(0, np.nan)
    vs3 = vs3.fillna(0.0)

    # 면적대비 에너지 사용비율 = 연면적 / 에너지 사용량
    upa = area_by_org / usage_cur.replace(0, np.nan)

    total_cur = float(usage_cur.sum())
    if total_cur == 0:
        share = pd.Series(0.0, index=usage_cur.index)
    else:
        share = usage_cur / total_cur

    df_org = pd.DataFrame(
        {
            "시설구분": fac_type_by_org,
            "연면적": area_by_org,
            "에너지 사용량": usage_cur,
            "면적대비 에너지 사용비율": upa,
            "에너지 사용 비중": share,
            "3개년 평균 에너지 사용량 대비 증감률": vs3,
        }
    )

    # 시설별 평균 면적 대비 에너지 사용비율
    facility_mean = df_org.groupby("시설구분", dropna=False)[
        "면적대비 에너지 사용비율"
    ].transform("mean")
    df_org["시설별 평균 면적 대비 에너지 사용비율"] = (
        df_org["면적대비 에너지 사용비율"]
        / facility_mean.replace(0, np.nan)
    )

    # 기관 고정 순서 적용
    df_org = df_org.reindex(org_order)
    baseline_cur = baseline_cur.reindex(org_order)

    return df_org, baseline_cur


def build_data_2_usage_analysis(
    year_to_raw: Mapping[int, pd.DataFrame],
    current_year: Optional[int] = None,
) -> Data2Result:
    """data_2: 공단 전체 + 소속기구별 에너지 사용량 분석."""
    spec = load_spec()
    df_all = _concat_raw(year_to_raw)

    if current_year is None:
        current_year = int(spec["meta"]["current_year"])

    s_overall_usage = _compute_overall_usage(df_all, spec, current_year)
    s_fac = _compute_overall_by_facility(df_all, current_year)
    s_all = pd.concat([s_overall_usage, s_fac])
    df_overall = s_all.to_frame().T
    df_overall.index = ["전 체"]

    df_by_org, baseline_cur = _compute_org_level_current_metrics(
        df_all, spec, current_year
    )

    return Data2Result(
        overall=df_overall,
        by_org=df_by_org,
        baseline_by_org=baseline_cur,
    )


# ===============================
# data_3 – 피드백
# ===============================
@dataclass
class Data3Result:
    overall: pd.DataFrame   # 공단 전체 피드백
    by_org: pd.DataFrame    # 소속기구별 피드백 요약
    detail: pd.DataFrame    # 관리대상 상세(O/X)


def _compute_overall_feedback(
    df_all: pd.DataFrame,
    spec: dict,
    current_year: int,
) -> pd.Series:
    """
    공단 전체 피드백 (권장 사용량 + 감축률) 계산.

    엑셀 규칙을 그대로 따른다.

    - 권장 에너지 사용량:
        기준연도(= current_year 바로 이전 연도)의 전체 사용량 합계 × (1 - NDC)
        → '1. 백데이터 분석'!U7 × (1 - NDC)에 해당

    - 전년대비 감축률:
        NDC 연평균 감축률 자체를 그대로 사용 (-NDC)

    - 3개년(과거 평균) 대비 감축률:
        (권장 사용량 - 기준연도 이전 모든 연도의 평균 합계) / 그 평균 합계
        → '3. 피드백'!에서 ('권장 사용량' - '1. 백데이터 분석'!U23) / U23 에 해당
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    # 연도별 전체 사용량 합계 (데이터가 없는 연도는 0으로)
    total_by_year = df_all.groupby("연도", dropna=False)["연단위"].sum()

    # spec 정의 + 실제 데이터 연도 모두 포함
    years_in_data = sorted(int(y) for y in total_by_year.index)
    years_all = sorted(set(analysis_years) | set(years_in_data))

    total_by_year = total_by_year.reindex(years_all, fill_value=0.0)

    if total_by_year.isna().any():
        raise ValueError("연도별 연단위 합계 계산 중 NaN 이 발생했습니다.")

    years_sorted = years_all

    if current_year not in years_sorted:
        raise ValueError(f"current_year={current_year} 가 분석 연도 목록에 없습니다.")

    idx = years_sorted.index(current_year)
    if idx == 0:
        raise ValueError(
            "current_year 앞에 기준이 될 전년 데이터가 없어 권장 사용량을 계산할 수 없습니다."
        )

    # 기준연도: current_year 바로 이전 연도 (엑셀 U7 에 해당)
    base_year = years_sorted[idx - 1]
    base_total = float(total_by_year.loc[base_year])

    # 권장 에너지 사용량 = 기준연도 전체 사용량 × (1 - NDC)
    recommended = base_total * (1.0 - ndc_rate)

    # 전년대비 감축률 = 정책 NDC 값 그대로
    reduction_yoy = -ndc_rate

    # 3개년(과거 평균) 대비 감축률
    #   기준연도 이전에 존재하는 모든 연도의 평균 합계 (엑셀 U23에 해당)
    past_years = years_sorted[: idx - 1]  # base_year 이전 연도들
    if past_years:
        past_avg = float(total_by_year.loc[past_years].mean())
    else:
        past_avg = base_total

    reduction_vs3 = (recommended - past_avg) / past_avg if past_avg != 0 else 0.0

    return pd.Series(
        {
            "권장 에너지 사용량": recommended,
            "전년대비 감축률": reduction_yoy,
            "3개년 대비 감축률": reduction_vs3,
        }
    )


def _compute_org_recommended_and_flags(
    df_org_metrics: pd.DataFrame,
    spec: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    소속기구별 권장 사용량 / 순위 / 관리대상 플래그, 상세표 생성.

    - 권장 에너지 사용량: 기관별 3개년 평균 × (1 - NDC)
      (3개년 평균은 vs3_rate 를 이용해 역산)
    - 관리대상 여부: (면적대비 과사용 OR 3개년 평균 대비 급증 OR 권장량 대비 과사용)
    """
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    cur_usage = df_org_metrics["에너지 사용량"]
    vs3_rate = df_org_metrics["3개년 평균 에너지 사용량 대비 증감률"]
    upa = df_org_metrics["면적대비 에너지 사용비율"]

    # avg3 역산: (cur - avg3) / avg3 = r  → cur = avg3 * (1 + r)
    avg3 = cur_usage / (1.0 + vs3_rate.replace(-1.0, np.nan))

    # 기관별 권장 사용량
    recommended = avg3 * (1.0 - ndc_rate)
    usage_vs_recommended = cur_usage / recommended.replace(0, np.nan)

    growth_rate = (cur_usage - avg3) / avg3.replace(0, np.nan)

    # 순위 (내림차순, 1등이 가장 큰 값)
    rank_by_usage = cur_usage.rank(ascending=False, method="min")
    rank_by_growth = growth_rate.rank(ascending=False, method="min")
    rank_by_upa = upa.rank(ascending=False, method="min")

    if (
        rank_by_usage.isna().any()
        or rank_by_growth.isna().any()
        or rank_by_upa.isna().any()
    ):
        _log_warning("일부 기관에서 순위 계산에 NaN 이 발생하여 순위를 0으로 표기합니다.")

    rank_by_usage_val = rank_by_usage.fillna(0.0)
    rank_by_growth_val = rank_by_growth.fillna(0.0)
    rank_by_upa_val = rank_by_upa.fillna(0.0)

    # 전체 평균값
    upa_mean = upa.mean()
    growth_mean = growth_rate.mean()
    uv_mean = usage_vs_recommended.mean()

    # 관리대상 조건
    cond_area = upa > upa_mean
    cond_growth = growth_rate > growth_mean
    cond_uv = usage_vs_recommended > uv_mean

    flag_series = (
        cond_area.fillna(False)
        | cond_growth.fillna(False)
        | cond_uv.fillna(False)
    ).astype(bool)
    flag_text = flag_series.map(lambda x: "O" if x else "X")

    df_by_org = pd.DataFrame(
        {
            "사용 분포 순위": rank_by_usage_val,
            "에너지 3개년 평균 증가 순위": rank_by_growth_val,
            "평균 에너지 사용량(연면적 기준) 순위": rank_by_upa_val,
            "권장 에너지 사용량": recommended,
            "권장 사용량 대비 에너지 사용 비율": usage_vs_recommended,
            "에너지 사용량 관리 대상": flag_text,
        },
        index=df_org_metrics.index,
    )

    # 상세 관리대상 표 – 각 조건을 그대로 노출
    detail_cols: Dict[str, pd.Series] = {}
    detail_cols["면적대비 에너지 과사용"] = cond_area
    detail_cols["에너지 사용량 급증(3개년 평균대비)"] = cond_growth
    detail_cols["권장량 대비 에너지 사용량 매우 초과"] = cond_uv

    df_detail = pd.DataFrame(index=df_org_metrics.index)
    for col_name, cond in detail_cols.items():
        cond_bool = cond.fillna(False).astype(bool)
        df_detail[col_name] = cond_bool.map(lambda x: "O" if x else "X")

    return df_by_org, df_detail


def build_data_3_feedback(
    year_to_raw: Mapping[int, pd.DataFrame],
    current_year: Optional[int] = None,
) -> Data3Result:
    """data_3: 공단 전체 / 소속기구별 피드백 + 관리대상 상세."""
    spec = load_spec()
    df_all = _concat_raw(year_to_raw)

    if current_year is None:
        current_year = int(spec["meta"]["current_year"])

    # data_2 와 동일 로직으로 기관별 지표 + 3개년 평균을 얻는다
    df_org_metrics, _baseline_by_org = _compute_org_level_current_metrics(
        df_all, spec, current_year
    )

    s_overall = _compute_overall_feedback(df_all, spec, current_year)
    df_overall = s_overall.to_frame().T
    df_overall.index = ["공단 전체"]

    df_by_org, df_detail = _compute_org_recommended_and_flags(
        df_org_metrics, spec
    )

    return Data3Result(overall=df_overall, by_org=df_by_org, detail=df_detail)


# -----------------------------------------------------------
# 기존 app 호환용 wrapper (소속기구별 피드백 표만 필요할 때)
# -----------------------------------------------------------
def compute_facility_feedback(
    selected_year: int,
    year_to_raw: Mapping[int, pd.DataFrame],
):
    """
    기존 app.py 에서 사용하던 인터페이스를 유지하기 위한 래퍼.
    - 반환: (소속기구별 피드백 요약, 관리대상 상세표)
    """
    result = build_data_3_feedback(year_to_raw, current_year=selected_year)
    return result.by_org, result.detail
