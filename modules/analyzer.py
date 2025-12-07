# modules/analyzer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # 테스트 환경용
    st = None  # type: ignore[assignment]

from .loader import load_spec, get_org_order


# ======================================================================
# 공통 유틸
# ======================================================================


def _log_error(msg: str) -> None:
    if st is not None:
        st.error(msg)
    else:
        print(f"[ERROR] {msg}")


def _log_warning(msg: str) -> None:
    if st is not None:
        st.warning(msg)
    else:
        print(f"[WARN] {msg}")


# ======================================================================
# df_raw 결합
# ======================================================================


def _concat_raw(year_to_raw: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """
    loader.load_energy_files 가 반환한 year_to_raw 를 하나의 df로 합친다.

    필수 컬럼 (loader.build_df_raw 기준):
      ['기관명', '시설구분', '연면적', '연단위', '연도']
    """
    if not year_to_raw:
        raise ValueError(
            "year_to_raw 가 비어 있습니다. 먼저 에너지 사용량 파일을 업로드해 주세요."
        )

    required_cols = ["기관명", "시설구분", "연면적", "연단위", "연도"]
    dfs: List[pd.DataFrame] = []

    for year, df in year_to_raw.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()

        # 1) 필수 컬럼 존재 여부 확인
        for col in required_cols:
            if col not in tmp.columns:
                raise ValueError(
                    f"{year}년 df_raw에 '{col}' 컬럼이 없습니다. "
                    "loader 단계에서 스키마를 확인해 주세요."
                )

        # 2) 숫자 컬럼 정제
        num_cols = ["연도", "연면적", "연단위"]
        for col in num_cols:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        # 3) 숫자 컬럼에 NaN 이 있는 행은 경고 후 분석에서 제외
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

    # 컬럼 형식 정리
    df_all["연도"] = df_all["연도"].astype(int)
    df_all["기관명"] = df_all["기관명"].astype(str)
    df_all["시설구분"] = df_all["시설구분"].astype(str)

    return df_all


# ======================================================================
# data_2. 에너지 사용량 분석
# ======================================================================


@dataclass
class Data2Result:
    overall: pd.DataFrame
    by_org: pd.DataFrame


def _compute_overall_usage(
    df_all: pd.DataFrame, spec: dict, current_year: int
) -> pd.Series:
    """
    overall_current_year_usage + overall_yoy_and_3yr_change 계산.
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]

    if current_year not in analysis_years:
        raise ValueError(
            f"current_year={current_year} 가 meta.analysis_years 에 없습니다."
        )

    prev_year = current_year - 1
    if prev_year not in analysis_years:
        raise ValueError(f"전년({prev_year}) 데이터가 meta.analysis_years 에 없습니다.")

    # 연도별 전체 사용량 합계
    total_by_year = (
        df_all.groupby("연도", dropna=False)["연단위"].sum().reindex(analysis_years)
    )

    if total_by_year.isna().any():
        raise ValueError("연도별 연단위 합계 계산 중 NaN 이 발생했습니다.")

    cur = float(total_by_year.loc[current_year])
    prev = float(total_by_year.loc[prev_year])

    # current_year 이전 연도 중에서 마지막 3개년
    past_years = [y for y in analysis_years if y < current_year]
    three_years = past_years[-3:]
    avg3 = float(total_by_year.loc[three_years].mean())

    if prev == 0 or avg3 == 0:
        raise ValueError("전년 또는 3개년 평균 사용량이 0입니다. 계산이 불가능합니다.")

    yoy_rate = (cur - prev) / prev
    vs3_rate = (cur - avg3) / avg3

    return pd.Series(
        {
            "에너지 사용량(현재 기준)": cur,
            "전년대비 증감률": yoy_rate,
            "3개년 평균 에너지 사용량 대비 증감률": vs3_rate,
        }
    )


def _compute_overall_by_facility(df_all: pd.DataFrame, current_year: int) -> pd.Series:
    """
    overall_usage_by_facility_type.usage_per_area(시설구분별) 계산.

    엑셀 기준에 맞추어:
      - 먼저 공단 전체 면적당 사용량(연단위/연면적)을 계산하고,
      - 각 시설군의 면적당 사용량을 이 값으로 나눈 '비율'을 반환한다.
    """
    df_year = df_all[df_all["연도"] == current_year].copy()
    if df_year.empty:
        raise ValueError(f"{current_year}년 데이터가 없습니다.")

    total_area = float(df_year["연면적"].sum())
    total_usage = float(df_year["연단위"].sum())
    if total_area <= 0 or total_usage < 0:
        raise ValueError("공단 전체 연면적 또는 에너지 사용량이 비정상입니다.")

    overall_upa = total_usage / total_area  # 공단 전체 면적당 사용량

    grp = df_year.groupby("시설구분", dropna=False).agg(
        usage_sum=("연단위", "sum"),
        area_sum=("연면적", "sum"),
    )

    if (grp["area_sum"] == 0).any():
        raise ValueError("시설구분별 연면적 합계가 0인 항목이 있습니다.")

    grp["usage_per_area_ratio"] = (
        (grp["usage_sum"] / grp["area_sum"]) / overall_upa
    )

    def get_value(ftype: str) -> float:
        if ftype not in grp.index:
            return float("nan")
        return float(grp.loc[ftype, "usage_per_area_ratio"])

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
) -> pd.DataFrame:
    """
    org_level_current_year_metrics 구현.

    반환 컬럼:
      ['시설구분','연면적','에너지 사용량','면적대비 에너지 사용비율',
       '에너지 사용 비중','3개년 평균 에너지 사용량 대비 증감률',
       '시설별 평균 면적 대비 에너지 사용비율']
    """
    # 사용 연도 집합: spec.logic.rules.calculations → year in 필터
    calc_conf = None
    for c in spec["logic"]["rules"]["calculations"]:
        if c.get("name") == "org_level_current_year_metrics":
            calc_conf = c
            break

    if calc_conf is None:
        raise ValueError(
            "spec.logic.rules.calculations 에 org_level_current_year_metrics 가 없습니다."
        )

    years_filter: List[int] = []
    for f in calc_conf.get("filters", []):
        if f.get("field") == "year" and f.get("op") == "in":
            years_filter = list(f.get("value", []))
            break

    if not years_filter:
        raise ValueError(
            "org_level_current_year_metrics 의 year in 필터를 spec 에서 찾을 수 없습니다."
        )

    # 실제 존재하는 연도만 사용
    years_filter = sorted(
        [y for y in years_filter if y in df_all["연도"].unique()]
    )
    if not years_filter:
        raise ValueError("year in 필터에 해당하는 데이터가 없습니다.")

    df = df_all[df_all["연도"].isin(years_filter)].copy()
    if df.empty:
        raise ValueError(f"year in {years_filter} 데이터가 없습니다.")

    analysis_years: List[int] = spec["meta"]["analysis_years"]
    analysis_years = [y for y in analysis_years if y in years_filter]

    # 연도별 연단위 합계 (소속기구별)
    usage_by_year_org = (
        df.groupby(["기관명", "연도"], dropna=False)["연단위"].sum().unstack("연도")
    )

    # 누락 연도는 0으로 채움
    for y in analysis_years:
        if y not in usage_by_year_org.columns:
            usage_by_year_org[y] = 0.0

    usage_by_year_org = usage_by_year_org[sorted(usage_by_year_org.columns)]

    # area, facility_type
    area_by_org = df.groupby("기관명", dropna=False)["연면적"].max()
    fac_type_by_org = df.groupby("기관명", dropna=False)["시설구분"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )

    # 현재연도 사용량
    if current_year not in usage_by_year_org.columns:
        usage_by_year_org[current_year] = 0.0
    usage_cur = usage_by_year_org[current_year]

    # 3개년 평균(현재연도 이전 최대 3개년, 기관별)
    past_years = [y for y in analysis_years if y < current_year]
    three_years = past_years[-3:]
    if three_years:
        avg3 = usage_by_year_org[three_years].mean(axis=1)
    else:
        # 과거 데이터가 없으면 현재값을 기준으로(증감률 0)
        avg3 = usage_cur.copy()

    # 증감률 (cur - avg3) / avg3
    vs3 = (usage_cur - avg3) / avg3.replace(0, np.nan)

    # 면적당 사용량
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

    # 시설별 평균 면적 대비 사용비율
    facility_mean = df_org.groupby("시설구분", dropna=False)[
        "면적대비 에너지 사용비율"
    ].transform("mean")
    df_org["시설별 평균 면적 대비 에너지 사용비율"] = (
        df_org["면적대비 에너지 사용비율"]
        / facility_mean.replace(0, np.nan)
    )

    # ⚠ 여기서는 reindex 하지 않고, 실제 존재하는 기관만 반환
    # (공단 전체/기관별 보기에서의 정렬/필터는 app.py 에서 처리)
    return df_org


def build_data_2_usage_analysis(
    year_to_raw: Mapping[int, pd.DataFrame],
    current_year: Optional[int] = None,
) -> Data2Result:
    """
    data_2. 에너지 사용량 분석 전체 계산.

    Returns
    -------
    Data2Result
        overall: 1. 공단 전체기준 (1행 6열)
        by_org: 2. 소속기구별 (기관 x 7열)
    """
    spec = load_spec()
    df_all = _concat_raw(year_to_raw)

    if current_year is None:
        current_year = int(spec["meta"]["current_year"])

    # 1) 공단 전체
    s_overall_usage = _compute_overall_usage(df_all, spec, current_year)
    s_fac = _compute_overall_by_facility(df_all, current_year)
    s_all = pd.concat([s_overall_usage, s_fac])
    df_overall = s_all.to_frame().T
    df_overall.index = ["전 체"]

    # 2) 소속기구별
    df_by_org = _compute_org_level_current_metrics(df_all, spec, current_year)

    return Data2Result(overall=df_overall, by_org=df_by_org)


# ======================================================================
# data_3. 피드백
# ======================================================================


@dataclass
class Data3Result:
    overall: pd.DataFrame
    by_org: pd.DataFrame
    detail: pd.DataFrame


def _compute_overall_feedback(
    df_all: pd.DataFrame,
    spec: dict,
    current_year: int,
) -> pd.Series:
    """
    overall_recommended_usage_by_ndc 계산.
    """
    analysis_years: List[int] = spec["meta"]["analysis_years"]
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    total_by_year = (
        df_all.groupby("연도", dropna=False)["연단위"].sum().reindex(analysis_years)
    )
    if total_by_year.isna().any():
        raise ValueError("연도별 연단위 합계 계산 중 NaN 이 발생했습니다.")

    cur = float(total_by_year.loc[current_year])

    # current_year 이전 연도 중 마지막 3개년
    past_years = [y for y in analysis_years if y < current_year]
    three_years = past_years[-3:]
    avg3 = float(total_by_year.loc[three_years].mean())

    if avg3 == 0:
        raise ValueError("3개년 평균 사용량이 0입니다.")

    recommended = cur * (1.0 - ndc_rate)
    reduction_yoy = -ndc_rate
    reduction_vs3 = (recommended - avg3) / avg3

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
    org_level_recommended_usage_and_flags + 상세 관리대상 표 생성.

    management_flag 규칙:
      - 조건1: usage_vs_recommended > 1
      - 조건2: usage_per_area > 전체 기관 평균 usage_per_area
      - 최종: 조건1 OR 조건2
    """
    ndc_rate: float = float(spec["meta"]["ndc_target_rate"])

    # df_org_metrics 는 _compute_org_level_current_metrics 결과
    cur_usage = df_org_metrics["에너지 사용량"]

    recommended = cur_usage * (1.0 - ndc_rate)
    usage_vs_recommended = cur_usage / recommended.replace(0, np.nan)

    # 3개년 평균 대비 성장률은 metrics 에서 계산된 값을 그대로 사용
    growth_rate = df_org_metrics["3개년 평균 에너지 사용량 대비 증감률"]

    # 순위 계산 (내림차순, 1등이 가장 큰 값) – float 그대로 유지
    rank_by_usage = cur_usage.rank(ascending=False, method="min")
    rank_by_growth = growth_rate.rank(ascending=False, method="min")
    rank_by_upa = df_org_metrics["면적대비 에너지 사용비율"].rank(
        ascending=False, method="min"
    )

    # NaN 순위는 0으로 표기
    if (
        rank_by_usage.isna().any()
        or rank_by_growth.isna().any()
        or rank_by_upa.isna().any()
    ):
        _log_warning("일부 기관에서 순위 계산에 NaN 이 발생하여 순위를 0으로 표기합니다.")

    rank_by_usage_val = rank_by_usage.fillna(0.0)
    rank_by_growth_val = rank_by_growth.fillna(0.0)
    rank_by_upa_val = rank_by_upa.fillna(0.0)

    # management_flag: 조건1 OR 조건2
    upa = df_org_metrics["면적대비 에너지 사용비율"]
    upa_mean = upa.mean()

    cond1 = usage_vs_recommended > 1.0
    cond2 = upa > upa_mean
    flag_series = (cond1.fillna(False) | cond2.fillna(False)).astype(bool)
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

    # 상세 관리대상 표
    detail_cols: Dict[str, pd.Series] = {}
    detail_cols["면적대비 에너지 과사용"] = upa > upa_mean
    detail_cols["에너지 사용량 급증(3개년 평균대비)"] = growth_rate > 0
    detail_cols["권장량 대비 에너지 사용량 매우 초과"] = usage_vs_recommended > 1.0

    df_detail = pd.DataFrame(index=df_org_metrics.index)
    for col_name, cond in detail_cols.items():
        cond_bool = cond.fillna(False).astype(bool)
        df_detail[col_name] = cond_bool.map(lambda x: "O" if x else "X")

    return df_by_org, df_detail


def build_data_3_feedback(
    year_to_raw: Mapping[int, pd.DataFrame],
    current_year: Optional[int] = None,
) -> Data3Result:
    """
    data_3. 피드백 전체 계산.

    Returns
    -------
    Data3Result
        overall: 1. 공단 전체기준
        by_org: 2. 소속기구별 요약
        detail: 3. 에너지 사용량 관리 대상 상세 (O/X 표)
    """
    spec = load_spec()
    df_all = _concat_raw(year_to_raw)

    if current_year is None:
        current_year = int(spec["meta"]["current_year"])

    # 1) 공단 전체
    s_overall = _compute_overall_feedback(df_all, spec, current_year)
    df_overall = s_overall.to_frame().T
    df_overall.index = ["공단 전체"]

    # 2) 소속기구별 metrics (data_2와 동일 로직 재사용)
    df_org_metrics = _compute_org_level_current_metrics(df_all, spec, current_year)

    # 3) 권장 사용량 / 관리대상 플래그 / 상세 표
    df_by_org, df_detail = _compute_org_recommended_and_flags(df_org_metrics, spec)

    return Data3Result(overall=df_overall, by_org=df_by_org, detail=df_detail)


# ======================================================================
# 기존 app 호환용 래퍼 (compute_facility_feedback 등)
# ======================================================================


def compute_facility_feedback(
    selected_year: int,
    year_to_raw: Mapping[int, pd.DataFrame],
):
    """
    기존 app.py 에서 사용 중인 인터페이스를 유지하기 위한 래퍼.
    """
    result = build_data_3_feedback(year_to_raw, current_year=selected_year)
    return result.by_org, result.detail
