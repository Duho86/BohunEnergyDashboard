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
    if not year_to_raw:
        raise ValueError("year_to_raw 가 비어 있습니다. 먼저 파일을 업로드해 주세요.")

    required_cols = ["기관명", "시설구분", "연면적", "연단위", "연도"]
    dfs: List[pd.DataFrame] = []

    for year, df in year_to_raw.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()

        # 엑셀 합계행(기관명='합계' & 연면적/연단위 NaN)은 미리 제거
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
    overall: pd.DataFrame
    by_org: pd.DataFrame
    baseline_by_org: pd.Series  # 각 기관별 3개년 평균 (엑셀 B7, U7에 해당)


def _compute_overall_usage(
    df_all: pd.DataFrame, spec: dict, current_year: int
) -> pd.Series:
    analysis_years: List[int] = spec["meta"]["analysis_years"]

    if current_year not in analysis_years:
        raise ValueError(
            f"current_year={current_year} 가 meta.analysis_years 에 없습니다."
        )

    prev_year = current_year - 1
    if prev_year not in analysis_years:
        raise ValueError(f"전년({prev_year}) 데이터가 meta.analysis_years 에 없습니다.")

    total_by_year = (
        df_all.groupby("연도", dropna=False)["연단위"].sum().reindex(analysis_years)
    )
    if total_by_year.isna().any():
        raise ValueError("연도별 연단위 합계 계산 중 NaN 이 발생했습니다.")

    cur = float(total_by_year.loc[current_year])
    prev = float(total_by_year.loc[prev_year])

    past_years = [y for y in analysis_years if y < current_year]
    three_years = past_years[-3:]
    avg3 = float(total_by_year.loc[three_years].mean())

    if prev == 0 or avg3 == 0:
        raise ValueError("전년 또는 3개년 평균 사용량이 0입니다.")

    yoy_rate = (cur - prev) / prev
    vs3_rate = (cur - avg3) / avg3

    return pd.Series(
        {
            "에너지 사용량(현재 기준)": cur,
            "전년대비 증감률": yoy_rate,
            "3개년 평균 에너지 사용량 대비 증감률": vs3_rate,
        }
    )


def _compute_overall_by_facility(
    df_all: pd.DataFrame, current_year: int
) -> pd.Series:
    """
    시설구분별 '면적대비 에너지 사용비율' 평균.

    ✔ 엑셀 정의:
      - 기관별 면적대비 비율 = 에너지 사용량 / 연면적
      - 시설구분별 평균 = 그 기관들의 단순 평균
    """
    df_year = df_all[df_all["연도"] == current_year].copy()
    if df_year.empty:
        raise ValueError(f"{current_year}년 데이터가 없습니다.")

    usage_by_org = df_year.groupby("기관명", dropna=False)["연단위"].sum()
    area_by_org = df_year.groupby("기관명", dropna=False)["연면적"].max()
    fac_type_by_org = df_year.groupby("기관명", dropna=False)["시설구분"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )

    # 사용량=0 또는 면적<=0 기관 제외
    valid_mask = (usage_by_org > 0) & (area_by_org > 0)
    if not valid_mask.all():
        _log_warning(
            "일부 기관에서 연면적 또는 에너지 사용량이 0 이하입니다. "
            "면적대비 사용비율 계산에서 제외합니다."
        )
        usage_by_org = usage_by_org[valid_mask]
        area_by_org = area_by_org[valid_mask]
        fac_type_by_org = fac_type_by_org[valid_mask]

    # ✅ 에너지 사용량 / 연면적
    upa_org = usage_by_org / area_by_org

    df_org = pd.DataFrame(
        {
            "시설구분": fac_type_by_org,
            "면적대비 에너지 사용비율": upa_org,
        }
    )

    grp = df_org.groupby("시설구분", dropna=False)["면적대비 에너지 사용비율"].mean()

    def get_value(ftype: str) -> float:
        return float(grp.get(ftype, np.nan))

    return pd.Series(
        {
            "의료시설": get_value("의료시설"),
            "복지시설": get_value("복지시설"),
            "기타시설": get_value("기타시설"),
        }
    )


def _compute_org_level_current_metrics(
    df_all: pd.DataFrame,
    spec: dict,
    current_year: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    org_level_current_year_metrics 구현.

    반환:
      - df_org: 기관별 현재연도 지표
      - baseline_cur: 기관별 3개년 평균 사용량 (엑셀 '1. 백데이터 분석'!B7 / U7 역할)
    """
    calc_conf = None
    for c in spec["logic"]["rules"]["calculations"]:
        if c.get("name") == "org_level_current_year_metrics":
            calc_conf = c
            break
    if calc_conf is None:
        raise ValueError("spec.logic.rules.calculations 에 org_level_current_year_metrics 없음")

    years_filter: List[int] = []
    for f in calc_conf.get("filters", []):
        if f.get("field") == "year" and f.get("op") == "in":
            years_filter = list(f.get("value", []))
            break
    if not years_filter:
        raise ValueError("org_level_current_year_metrics 의 year in 필터를 spec 에서 찾을 수 없습니다.")

    df = df_all[df_all["연도"].isin(years_filter)].copy()
    if df.empty:
        raise ValueError(f"year in {years_filter} 데이터가 없습니다.")

    years = sorted([y for y in years_filter if y in df["연도"].unique()])

    usage_by_year_org = (
        df.groupby(["기관명", "연도"], dropna=False)["연단위"].sum().unstack("연도")
    )
    for y in years:
        if y not in usage_by_year_org.columns:
            usage_by_year_org[y] = 0.0
    usage_by_year_org = usage_by_year_org[years]

    area_by_org = df.groupby("기관명", dropna=False)["연면적"].max()
    fac_type_by_org = df.groupby("기관명", dropna=False)["시설구분"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )

    if current_year not in usage_by_year_org.columns:
        usage_by_year_org[current_year] = 0.0
    usage_cur = usage_by_year_org[current_year]

    # 🔹 기관별 3개년 평균 (엑셀 B7 / U7)
    baseline_by_year_org = pd.DataFrame(
        index=usage_by_year_org.index, columns=years, dtype=float
    )
    for i, y in enumerate(years):
        prev_years = years[:i][-3:]
        if prev_years:
            baseline_by_year_org[y] = usage_by_year_org[prev_years].mean(axis=1)
        else:
            baseline_by_year_org[y] = usage_by_year_org[y]
    if current_year not in baseline_by_year_org.columns:
        raise ValueError(f"{current_year}년의 3개년 평균 기준을 계산할 수 없습니다.")

    baseline_cur = baseline_by_year_org[current_year]

    # 3개년 평균 대비 증감률
    vs3 = (usage_cur - baseline_cur) / baseline_cur.replace(0, np.nan)

    # ✅ 면적대비 에너지 사용비율 = 에너지 사용량 / 연면적
    upa = usage_cur / area_by_org.replace(0, np.nan)

    total_cur = float(usage_cur.sum())
    if total_cur == 0:
        raise ValueError("현재연도 전체 사용량 합계가 0입니다.")

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

    # 시설별 평균 대비 비율
    facility_mean = df_org.groupby("시설구분", dropna=False)[
        "면적대비 에너지 사용비율"
    ].transform("mean")
    df_org["시설별 평균 면적 대비 에너지 사용비율"] = (
        df_org["면적대비 에너지 사용비율"] / facility_mean.replace(0, np.nan)
    )

    return df_org, baseline_cur


def build_data_2_usage_analysis(
    year_to_raw: Mapping[int, pd.DataFrame],
    current_year: Optional[int] = None,
) -> Data2Result:
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

    return Data2Result(overall=df_overall, by_org=df_by_org, baseline_by_org=baseline_cur)


# ===============================
# data_3 – 피드백
# ===============================
@dataclass
class Data3Result:
    overall: pd.DataFrame
    by_org: pd.DataFrame
    detail: pd.DataFrame


def _compute_overall_feedback(
    df_all: pd.DataFrame,
    spec: dict,
    current_year: int,
    baseline_by_org: pd.Series,
) -> pd.Series:
    """
    공단 전체 피드백.

    엑셀 로직에 맞춰:
      - 권장 에너지 사용량 = (공단 전체 3개년 평균) × (1 - NDC)
         → baseline_by_org 합계 × (1 - ndc_rate)
      - 전년대비 감축률 = -NDC (목표치)
      - 3개년 대비 감축률 = (권장량 - 3개년 평균 전체)/3개년 평균 전체
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    total_by_year = (
        df_all.groupby("연도", dropna=False)["연단위"].sum().reindex(analysis_years)
    )
    if total_by_year.isna().any():
        raise ValueError("연도별 연단위 합계 계산 중 NaN 이 발생했습니다.")

    # 공단 전체 3개년 평균 (엑셀 U7)
    baseline_total = float(baseline_by_org.sum())

    # 최근 3개년 전체 평균 (엑셀 U23 유사 – 완전 동일하지 않을 수 있지만 최대한 근접)
    past_years = [y for y in analysis_years if y < current_year]
    three_years = past_years[-3:]
    baseline_total_ref = float(total_by_year.loc[three_years].mean())

    recommended = baseline_total * (1.0 - ndc_rate)
    if baseline_total_ref == 0:
        vs3_reduction = np.nan
    else:
        vs3_reduction = (recommended - baseline_total_ref) / baseline_total_ref

    reduction_yoy = -ndc_rate

    return pd.Series(
        {
            "권장 에너지 사용량": recommended,
            "전년대비 감축률": reduction_yoy,
            "3개년 대비 감축률": vs3_reduction,
        }
    )


def _compute_org_recommended_and_flags(
    df_org_metrics: pd.DataFrame,
    baseline_by_org: pd.Series,
    spec: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    org_level_recommended_usage_and_flags + 상세 관리대상 표.

    ✔ 권장 에너지 사용량:
       - 엑셀: '1. 백데이터 분석'!B7 * (1 - 4.17%)
       - 여기서는 baseline_by_org * (1 - ndc_rate)
    """
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    cur_usage = df_org_metrics["에너지 사용량"]
    share = df_org_metrics["에너지 사용 비중"]
    upa = df_org_metrics["면적대비 에너지 사용비율"]
    growth_rate = df_org_metrics["3개년 평균 에너지 사용량 대비 증감률"]

    # ✅ 기관별 권장 사용량 = 3개년 평균 × (1 - ndc_rate)
    recommended = baseline_by_org * (1.0 - ndc_rate)
    usage_vs_recommended = cur_usage / recommended.replace(0, np.nan)

    rank_by_usage = share.rank(ascending=False, method="average")
    rank_by_growth = growth_rate.rank(ascending=False, method="average")
    rank_by_upa = upa.rank(ascending=False, method="average")

    if (
        rank_by_usage.isna().any()
        or rank_by_growth.isna().any()
        or rank_by_upa.isna().any()
    ):
        _log_warning("일부 기관에서 순위 계산에 NaN 이 발생하여 순위를 0으로 표기합니다.")

    rank_by_usage_val = rank_by_usage.fillna(0.0)
    rank_by_growth_val = rank_by_growth.fillna(0.0)
    rank_by_upa_val = rank_by_upa.fillna(0.0)

    upa_mean_overall = float(upa.mean())
    growth_mean_overall = float(growth_rate.mean())
    uv_mean_overall = float(usage_vs_recommended.mean())

    cond_area = upa > upa_mean_overall
    cond_growth = growth_rate > growth_mean_overall
    cond_uv = usage_vs_recommended > uv_mean_overall

    management_flag = (cond_area | cond_growth | cond_uv).fillna(False)
    flag_text = management_flag.map(lambda x: "O" if x else "X")

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

    df_detail = pd.DataFrame(index=df_org_metrics.index)
    df_detail["면적대비 에너지 과사용"] = cond_area.fillna(False).map(
        lambda x: "O" if x else "X"
    )
    df_detail["에너지 사용량 급증(3개년 평균대비)"] = cond_growth.fillna(False).map(
        lambda x: "O" if x else "X"
    )
    df_detail["권장량 대비 에너지 사용량 매우 초과"] = cond_uv.fillna(False).map(
        lambda x: "O" if x else "X"
    )

    return df_by_org, df_detail


def build_data_3_feedback(
    year_to_raw: Mapping[int, pd.DataFrame],
    current_year: Optional[int] = None,
) -> Data3Result:
    spec = load_spec()
    df_all = _concat_raw(year_to_raw)

    if current_year is None:
        current_year = int(spec["meta"]["current_year"])

    # data_2와 동일 로직으로 기관별 지표 + 3개년 평균을 얻는다
    df_org_metrics, baseline_by_org = _compute_org_level_current_metrics(
        df_all, spec, current_year
    )

    s_overall = _compute_overall_feedback(df_all, spec, current_year, baseline_by_org)
    df_overall = s_overall.to_frame().T
    df_overall.index = ["공단 전체"]

    df_by_org, df_detail = _compute_org_recommended_and_flags(
        df_org_metrics, baseline_by_org, spec
    )

    return Data3Result(overall=df_overall, by_org=df_by_org, detail=df_detail)


# 기존 app 호환용
def compute_facility_feedback(
    selected_year: int,
    year_to_raw: Mapping[int, pd.DataFrame],
):
    result = build_data_3_feedback(year_to_raw, current_year=selected_year)
    return result.by_org, result.detail
