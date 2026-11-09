# modules/analyzer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

from .loader import load_spec, get_org_order


# ===============================
# 기본 정책·환산계수 (Secrets 미입력 시 NONE / 산정대기 방지용 Fallback)
# ===============================
DEFAULT_ENERGY_SETTINGS: Dict[str, Any] = {
    "energy_kwh_per_unit": {
        "전기|kWh": 1.0,
        "전기|MWh": 1000.0,
        "도시가스|m3": 10.55,
        "도시가스|Nm3": 10.55,
        "도시가스|천Nm3": 10550.0,
        "가스(LNG)|m3": 10.55,
        "가스(LNG)|Nm3": 10.55,
        "가스(LNG)|천Nm3": 10550.0,
        "가스(LNG)|MJ": 0.2777777778,
        "가스(LNG)|Gcal": 1162.2222222222,
        "가스(LPG)|kg": 13.99,
        "가스(LPG)|m3": 26.0,
        "가스(LPG)|L": 7.56,
        "지역난방|Gcal": 1162.2222222222,
        "지역난방|MWh": 1000.0,
        "지역난방|kWh": 1.0,
        "유류|L": 9.88,
        "유류(경유/등유)|L": 9.88,
        "경유|L": 9.88,
        "등유|L": 9.55,
        "상수도|m3": 0.65,
        "태양광|kWh": 1.0,
        "태양광|MWh": 1000.0,
        "지열|kWh": 1.0,
        "지열|MWh": 1000.0,
        "지열|Gcal": 1162.2222222222,
    },
    "emission_tco2_per_unit": {
        "전기|kWh": 0.0004781,
        "전기|MWh": 0.4781,
        "도시가스|m3": 0.002176,
        "도시가스|Nm3": 0.002176,
        "도시가스|천Nm3": 2.176,
        "가스(LNG)|m3": 0.002176,
        "가스(LNG)|Nm3": 0.002176,
        "가스(LNG)|천Nm3": 2.176,
        "가스(LNG)|MJ": 0.0000561,
        "가스(LNG)|Gcal": 0.2348,
        "가스(LPG)|kg": 0.00301,
        "가스(LPG)|m3": 0.00560,
        "가스(LPG)|L": 0.00162,
        "지역난방|Gcal": 0.0387,
        "지역난방|MWh": 0.0333,
        "지역난방|kWh": 0.0000333,
        "유류|L": 0.00258,
        "유류(경유/등유)|L": 0.00258,
        "경유|L": 0.00258,
        "등유|L": 0.00254,
        "상수도|m3": 0.000237,
    },
    "avoided_tco2_per_kwh": {
        "태양광": 0.0004781,
        "지열": 0.0001960,
    },
    "self_use_ratio": {},
    "consumption_basis": {},
    "capacity_unit": {},
    "by_year": {},
}

FALLBACK_KWH_BY_FUEL: Dict[str, float] = {
    "전기": 1.0,
    "도시가스": 10.55,
    "가스(LNG)": 10.55,
    "가스(LPG)": 13.99,
    "지역난방": 1162.2222222222,
    "유류": 9.88,
    "유류(경유/등유)": 9.88,
    "경유": 9.88,
    "등유": 9.55,
    "상수도": 0.65,
    "태양광": 1.0,
    "지열": 1.0,
}

FALLBACK_EMISSION_TCO2_PER_KWH: Dict[str, float] = {
    "전기": 0.0004781,
    "도시가스": 0.00020625,
    "가스(LNG)": 0.00020625,
    "가스(LPG)": 0.00021515,
    "지역난방": 0.0000333,
    "유류": 0.0002611,
    "유류(경유/등유)": 0.0002611,
    "경유": 0.0002611,
    "등유": 0.0002659,
    "상수도": 0.0003646,
}

FALLBACK_AVOIDED_TCO2_PER_KWH: Dict[str, float] = {
    "태양광": 0.0004781,
    "지열": 0.0001960,
}


def get_default_energy_settings() -> Dict[str, Any]:
    """
    gemini-code-app.py 호환 및 Streamlit Secrets 미입력 환경에서
    즉시 기본 에너지·온실가스 환산계수 설정 사전을 반환한다.
    """
    return {
        k: (dict(v) if isinstance(v, Mapping) else v)
        for k, v in DEFAULT_ENERGY_SETTINGS.items()
    }


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
    - 연도/연면적/연단위에 NaN 이 있는 행은 안전하게 보정 후 병합한다.
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

        # 숫자 컬럼 정제 (신재생 대시보드 등에서 연면적 결측 시 1.0으로 보정하여 제외 방지)
        tmp["연도"] = pd.to_numeric(tmp["연도"], errors="coerce").fillna(int(year))
        tmp["연면적"] = pd.to_numeric(tmp["연면적"], errors="coerce").fillna(1.0)
        tmp["연단위"] = pd.to_numeric(tmp["연단위"], errors="coerce").fillna(0.0)

        na_mask = tmp[["연도", "연면적", "연단위"]].isna().any(axis=1)
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
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]

    years_in_data = sorted(int(y) for y in df_all["연도"].dropna().unique())
    if not years_in_data:
        raise ValueError("df_raw 에 연도 데이터가 없습니다.")

    years_all = sorted(set(analysis_years) | set(years_in_data))

    if current_year not in years_all:
        years_all = sorted(set(years_all) | {current_year})

    total_by_year = df_all.groupby("연도", dropna=False)["연단위"].sum()
    total_by_year = total_by_year.reindex(years_all, fill_value=0.0)

    cur = float(total_by_year.loc[current_year])

    prev_candidates = [y for y in years_all if y < current_year]
    if prev_candidates:
        prev_year = prev_candidates[-1]
        prev = float(total_by_year.loc[prev_year])
    else:
        prev = 0.0

    yoy_rate = (cur - prev) / prev if prev != 0 else 0.0

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
    """
    df_year = df_all[df_all["연도"] == current_year].copy()
    if df_year.empty:
        return pd.Series(
            {"의료시설": 0.0, "복지시설": 0.0, "기타시설": 0.0}
        )

    grouped = (
        df_year.groupby(["기관명", "시설구분"], dropna=False)[["연단위", "연면적"]]
        .agg({"연단위": "sum", "연면적": "max"})
    )

    area_per_usage = grouped["연면적"] / grouped["연단위"].replace(0, np.nan)
    grouped["면적대비 에너지 사용비율"] = area_per_usage

    fac_mean = (
        grouped.groupby("시설구분", dropna=False)["면적대비 에너지 사용비율"]
        .mean()
    )

    return pd.Series(
        {
            "의료시설": float(fac_mean.get("의료시설", 0.0) if pd.notna(fac_mean.get("의료시설")) else 0.0),
            "복지시설": float(fac_mean.get("복지시설", 0.0) if pd.notna(fac_mean.get("복지시설")) else 0.0),
            "기타시설": float(fac_mean.get("기타시설", 0.0) if pd.notna(fac_mean.get("기타시설")) else 0.0),
        }
    )


def _compute_org_level_current_metrics(
    df_all: pd.DataFrame,
    spec: dict,
    current_year: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    2. 에너지 사용량 분석 – 소속기구별 표 전체를 계산한다.
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]

    years_in_data = sorted(int(y) for y in df_all["연도"].dropna().unique())
    years_all = sorted(set(analysis_years) | set(years_in_data) | {current_year})
    years = [y for y in years_all if (y in df_all["연도"].unique() or y == current_year)]

    usage_by_year_org = (
        df_all.groupby(["기관명", "연도"], dropna=False)["연단위"]
        .sum()
        .unstack("연도", fill_value=0.0)
        .reindex(columns=years, fill_value=0.0)
    )

    df_cur = df_all[df_all["연도"] == current_year].copy()
    if df_cur.empty:
        area_by_org = df_all.groupby("기관명", dropna=False)["연면적"].max()
        fac_type_by_org = df_all.groupby("기관명", dropna=False)["시설구분"].first()
    else:
        area_by_org = df_cur.groupby("기관명", dropna=False)["연면적"].max()
        fac_type_by_org = df_cur.groupby("기관명", dropna=False)["시설구분"].first()

    org_order = list(get_org_order())
    usage_by_year_org = usage_by_year_org.reindex(org_order, fill_value=0.0)
    area_by_org = area_by_org.reindex(org_order).fillna(0.0)
    fac_type_by_org = fac_type_by_org.reindex(org_order).fillna("기타시설")

    baseline_by_year_org = pd.DataFrame(
        index=usage_by_year_org.index, columns=years, dtype=float
    )
    for i, y in enumerate(years):
        prev_years = years[:i][-3:]
        if prev_years:
            baseline_by_year_org[y] = usage_by_year_org[prev_years].mean(axis=1)
        else:
            baseline_by_year_org[y] = usage_by_year_org[y]

    usage_cur = usage_by_year_org[current_year]
    baseline_cur = baseline_by_year_org[current_year]

    vs3 = (usage_cur - baseline_cur) / baseline_cur.replace(0, np.nan)
    vs3 = vs3.fillna(0.0)

    upa = (area_by_org / usage_cur.replace(0, np.nan)).fillna(0.0)

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

    facility_mean = df_org.groupby("시설구분", dropna=False)[
        "면적대비 에너지 사용비율"
    ].transform("mean")
    df_org["시설별 평균 면적 대비 에너지 사용비율"] = (
        df_org["면적대비 에너지 사용비율"]
        / facility_mean.replace(0, np.nan)
    ).fillna(0.0)

    df_org = df_org.reindex(org_order)
    baseline_cur = baseline_cur.reindex(org_order).fillna(0.0)

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
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    total_by_year = df_all.groupby("연도", dropna=False)["연단위"].sum()

    years_in_data = sorted(int(y) for y in total_by_year.index)
    years_all = sorted(set(analysis_years) | set(years_in_data) | {current_year})

    total_by_year = total_by_year.reindex(years_all, fill_value=0.0)
    years_sorted = years_all

    idx = years_sorted.index(current_year)
    if idx == 0:
        base_total = float(total_by_year.loc[current_year])
    else:
        base_year = years_sorted[idx - 1]
        base_total = float(total_by_year.loc[base_year])
        if base_total == 0.0:
            base_total = float(total_by_year.loc[current_year])

    recommended = base_total * (1.0 - ndc_rate)
    reduction_yoy = -ndc_rate

    past_years = years_sorted[: max(0, idx - 1)]
    if past_years:
        past_avg = float(total_by_year.loc[past_years].mean())
        if past_avg == 0.0:
            past_avg = base_total
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
    """
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    cur_usage = df_org_metrics["에너지 사용량"]
    vs3_rate = df_org_metrics["3개년 평균 에너지 사용량 대비 증감률"]
    upa = df_org_metrics["면적대비 에너지 사용비율"]

    avg3 = (cur_usage / (1.0 + vs3_rate.replace(-1.0, np.nan))).fillna(cur_usage)

    recommended = (avg3 * (1.0 - ndc_rate)).fillna(0.0)
    usage_vs_recommended = (cur_usage / recommended.replace(0, np.nan)).fillna(0.0)

    growth_rate = ((cur_usage - avg3) / avg3.replace(0, np.nan)).fillna(0.0)

    rank_by_usage = cur_usage.rank(ascending=False, method="min").fillna(0.0)
    rank_by_growth = growth_rate.rank(ascending=False, method="min").fillna(0.0)
    rank_by_upa = upa.rank(ascending=False, method="min").fillna(0.0)

    upa_mean = upa.mean() if not upa.dropna().empty else 0.0
    growth_mean = growth_rate.mean() if not growth_rate.dropna().empty else 0.0
    uv_mean = usage_vs_recommended.mean() if not usage_vs_recommended.dropna().empty else 0.0

    cond_area = (upa > upa_mean) & (cur_usage > 0)
    cond_growth = (growth_rate > growth_mean) & (cur_usage > 0)
    cond_uv = (usage_vs_recommended > uv_mean) & (cur_usage > 0)

    flag_series = (
        cond_area.fillna(False)
        | cond_growth.fillna(False)
        | cond_uv.fillna(False)
    ).astype(bool)
    flag_text = flag_series.map(lambda x: "O" if x else "X")

    df_by_org = pd.DataFrame(
        {
            "사용 분포 순위": rank_by_usage,
            "에너지 3개년 평균 증가 순위": rank_by_growth,
            "평균 에너지 사용량(연면적 기준) 순위": rank_by_upa,
            "권장 에너지 사용량": recommended,
            "권장 사용량 대비 에너지 사용 비율": usage_vs_recommended,
            "에너지 사용량 관리 대상": flag_text,
        },
        index=df_org_metrics.index,
    )

    detail_cols: Dict[str, pd.Series] = {
        "면적대비 에너지 과사용": cond_area,
        "에너지 사용량 급증(3개년 평균대비)": cond_growth,
        "권장량 대비 에너지 사용량 매우 초과": cond_uv,
    }

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


def compute_facility_feedback(
    selected_year: int,
    year_to_raw: Mapping[int, pd.DataFrame],
):
    """
    기존 app.py 에서 사용하던 인터페이스를 유지하기 위한 래퍼.
    """
    result = build_data_3_feedback(year_to_raw, current_year=selected_year)
    return result.by_org, result.detail


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

    usage = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연단위",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=years)
        .reindex(columns=org_order, fill_value=0)
    )
    usage["합계"] = usage.sum(axis=1)

    area = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연면적",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=years)
        .reindex(columns=org_order, fill_value=0)
    )
    area["합계"] = area.sum(axis=1)

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


def build_datal_tables(df_raw_all: pd.DataFrame):
    """구버전 코드 호환용 래퍼."""
    return build_data1_tables(df_raw_all)


# ===========================================================
# 소비 / 생산 대시보드 공통 분석 (Secrets 미설정 시 NONE/산정대기 방지 내장)
# ===========================================================
def settings_for_year(settings: Optional[Mapping[str, Any]], year: int) -> Dict[str, Any]:
    """연도별 정책계수 재정의. 내장 기본값(DEFAULT_ENERGY_SETTINGS)과 병합하여 결측을 원천 방지한다."""
    resolved: Dict[str, Any] = get_default_energy_settings()
    if settings:
        for k, v in settings.items():
            if isinstance(v, Mapping) and isinstance(resolved.get(k), Mapping):
                resolved[k] = {**resolved[k], **v}
            elif v is not None:
                resolved[k] = v

        year_overrides = settings.get("by_year", {}).get(str(year), {})
        for key, value in year_overrides.items():
            if isinstance(value, Mapping) and isinstance(resolved.get(key), Mapping):
                resolved[key] = {**resolved.get(key, {}), **value}
            elif value is not None:
                resolved[key] = value
    return resolved


def _factor_map(keys: pd.Series, values: Mapping[str, Any]) -> pd.Series:
    mapped = pd.to_numeric(keys.map(values), errors="coerce")
    if (mapped.dropna() < 0).any() or np.isinf(mapped.dropna()).any():
        raise ValueError("환산계수는 유한한 0 이상의 숫자여야 합니다.")
    return mapped


def prepare_energy_metrics(df: pd.DataFrame, settings: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    소비 및 생산 데이터의 에너지_kWh, 배출량_tCO2eq, 감축추정_tCO2eq, 확인용량_kW를 산정한다.
    Streamlit Secrets에 별도 계수가 없어도 내장 환산계수를 적용해 NONE / 산정대기를 방지한다.
    """
    out = df.copy()
    for column in ["수량", "설비용량_원본"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).to_numpy(
                dtype="float64"
            )
        else:
            out[column] = 0.0

    for column in [
        "에너지_kWh", "배출량_tCO2eq", "감축추정_tCO2eq", "확인용량_kW",
    ]:
        out[column] = 0.0

    for year, group in out.groupby("연도"):
        config = settings_for_year(settings, int(year))
        fuel_clean = group["에너지원"].astype(str).str.strip()
        unit_clean = group["단위"].astype(str).str.strip()
        keys = fuel_clean + "|" + unit_clean

        unit_factor = unit_clean.map({
            "kWh": 1.0,
            "MWh": 1000.0,
            "Gcal": 1162.2222222222222,
            "MJ": 0.2777777778,
            "m3": 10.55,
            "Nm3": 10.55,
            "천Nm3": 10550.0,
            "L": 9.88,
            "kg": 13.99,
        })
        configured_kwh = _factor_map(keys, config.get("energy_kwh_per_unit", {}))
        fuel_fallback_kwh = fuel_clean.map(FALLBACK_KWH_BY_FUEL).fillna(1.0)
        energy_factor = configured_kwh.combine_first(unit_factor).combine_first(fuel_fallback_kwh).fillna(1.0)

        energy = group["수량"] * energy_factor
        out.loc[group.index, "에너지_kWh"] = energy.to_numpy(dtype="float64", na_value=0.0)

        emissions_unit = _factor_map(keys, config.get("emission_tco2_per_unit", {}))
        emissions_from_kwh = energy * fuel_clean.map(FALLBACK_EMISSION_TCO2_PER_KWH).fillna(0.0004781)
        emissions_calc = (group["수량"] * emissions_unit).combine_first(emissions_from_kwh).fillna(0.0)

        out.loc[group.index, "배출량_tCO2eq"] = (
            emissions_calc.where(group["구분"].eq("소비"), 0.0).to_numpy(dtype="float64", na_value=0.0)
        )

        avoided_factor = _factor_map(
            fuel_clean, config.get("avoided_tco2_per_kwh", {})
        ).combine_first(fuel_clean.map(FALLBACK_AVOIDED_TCO2_PER_KWH)).fillna(0.0004781)

        out.loc[group.index, "감축추정_tCO2eq"] = (
            (energy * avoided_factor).where(group["구분"].eq("생산"), 0.0).to_numpy(dtype="float64", na_value=0.0)
        )

        capacity_keys = group["기관명"].astype(str) + "|" + fuel_clean
        units = capacity_keys.map(config.get("capacity_unit", {}))
        valid_configured = (
            (fuel_clean.eq("태양광") & units.eq("kW"))
            | (fuel_clean.eq("지열") & units.isin(["kWth", "kW"]))
        )
        valid_fallback = fuel_clean.isin(["태양광", "지열"]) & units.isna()
        valid = valid_configured | valid_fallback

        out.loc[group.index, "확인용량_kW"] = group["설비용량_원본"].where(
            valid & group["설비용량_원본"].gt(0), 0.0
        ).to_numpy(dtype="float64", na_value=0.0)

    return out


def complete_total(df: pd.DataFrame, column: str) -> float:
    """결측치 존재 시에도 NONE이나 산정대기가 발생하지 않도록 유효 합계를 반환한다."""
    if df is None or df.empty or column not in df.columns:
        return 0.0
    valid_series = pd.to_numeric(df[column], errors="coerce").dropna()
    if valid_series.empty:
        return 0.0
    return float(valid_series.sum())


def select_period(df: pd.DataFrame, year: int, end_month: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df["연도"].eq(year) & df["월"].le(end_month)].copy()


def period_summary(df: pd.DataFrame, year: int, end_month: int):
    selected = select_period(df, year, end_month)
    previous = select_period(df, year - 1, end_month)
    total = complete_total(selected, "에너지_kWh")
    prev_total = complete_total(previous, "에너지_kWh")

    yoy = (
        (total / prev_total - 1.0) * 100.0
        if pd.notna(total) and pd.notna(prev_total) and prev_total > 0
        else 0.0
    )

    org_order = list(get_org_order())
    org_rank_map = {org: idx for idx, org in enumerate(org_order)}

    rows = []
    if not selected.empty:
        for (org, fuel), group in selected.groupby(["기관명", "에너지원"], dropna=False):
            amount = complete_total(group, "에너지_kWh")
            equipment = group.drop_duplicates(["연도", "원본행"]) if "원본행" in group.columns else group
            capacities = pd.to_numeric(equipment["확인용량_kW"], errors="coerce").dropna()
            capacity = float(capacities.sum()) if not capacities.empty and capacities.sum() > 0 else 0.0
            eq_hours = (amount / capacity) if capacity > 0 else 0.0
            rows.append({
                "기관명": org,
                "에너지원": fuel,
                "에너지_kWh": amount,
                "기여도_%": (amount / total * 100.0) if total > 0 else 0.0,
                "확인용량_kW": capacity,
                "등가운전시간_h": eq_hours,
                "_org_order": org_rank_map.get(str(org), 999),
            })

    ranking = pd.DataFrame(rows, columns=[
        "기관명", "에너지원", "에너지_kWh", "기여도_%",
        "확인용량_kW", "등가운전시간_h", "_org_order"
    ])
    if not ranking.empty:
        ranking = (
            ranking.sort_values(["_org_order", "에너지_kWh"], ascending=[True, False])
            .drop(columns=["_org_order"])
            .reset_index(drop=True)
        )
    else:
        ranking = pd.DataFrame(columns=[
            "기관명", "에너지원", "에너지_kWh", "기여도_%",
            "확인용량_kW", "등가운전시간_h",
        ])

    return selected, total, yoy, ranking


def integrated_electricity(
    consumption: pd.DataFrame,
    production: pd.DataFrame,
    year: int,
    end_month: int,
    settings: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    config = settings_for_year(settings, year)
    current_c = select_period(consumption, year, end_month)
    current_p = select_period(production, year, end_month)

    org_order = list(get_org_order())
    present_orgs = [org for org in org_order if not current_c.empty and org in current_c["기관명"].unique()]
    if not current_c.empty:
        for org in current_c["기관명"].unique():
            if org not in present_orgs:
                present_orgs.append(org)

    rows = []
    for org in present_orgs:
        group = current_c[current_c["기관명"].eq(org)]
        electric = complete_total(group[group["에너지원"].eq("전기")], "에너지_kWh")
        solar_rows = (
            current_p[current_p["기관명"].eq(org) & current_p["에너지원"].eq("태양광")]
            if not current_p.empty else pd.DataFrame()
        )
        solar = complete_total(solar_rows, "에너지_kWh")

        ratio = config.get("self_use_ratio", {}).get(f"{org}|태양광", 1.0)
        if ratio is None or not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or not (0 <= ratio <= 1):
            ratio = 1.0

        self_used = solar * float(ratio)
        basis = config.get("consumption_basis", {}).get(org, "purchased")

        if basis == "gross" and self_used <= electric:
            net = electric - self_used
            status = "총수요에서 자가소비 차감"
        else:
            net = electric
            status = "구매전력 기준"

        rows.append({
            "기관명": org,
            "전력소비_입력_kWh": electric,
            "태양광_자가소비_kWh": self_used,
            "외부전력_필요량_kWh": net,
            "계량기준": status,
        })
    return pd.DataFrame(rows)
