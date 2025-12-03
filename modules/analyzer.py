# modules/analyzer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def _ensure_not_empty(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        raise ValueError(f"{name} 데이터가 비어 있습니다. 먼저 에너지 사용량 파일을 업로드해 주세요.")


def _force_numeric(s: pd.Series, col_name: str) -> pd.Series:
    """
    집계에 사용하는 컬럼을 무조건 float 로 강제 변환.

    - 콤마/공백 등이 섞여 있어도 pd.to_numeric(errors="coerce")로 숫자만 취함
    - 변환 실패 값은 NaN
    """
    try:
        s_num = pd.to_numeric(s, errors="coerce")
        return s_num.astype("float64")
    except Exception as e:
        raise RuntimeError(f"[숫자 변환] '{col_name}' 컬럼을 숫자로 변환하는 중 오류: {e}") from e


# 1) 월별·기관별 온실가스 환산량 계산
def get_monthly_ghg(
    df_std: pd.DataFrame,
    *,
    by_agency: bool = True,
    include_total: bool = False,
) -> pd.DataFrame:
    _ensure_not_empty(df_std, "표준 스키마")

    required = {"연도", "기관명", "월", "온실가스 환산량"}
    missing = required - set(df_std.columns)
    if missing:
        raise ValueError(f"[월별 집계 단계] 필수 컬럼 누락: {missing}")

    try:
        df = df_std.copy()

        df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
        df["월"] = pd.to_numeric(df["월"], errors="coerce").astype("Int64")
        df["온실가스 환산량"] = _force_numeric(df["온실가스 환산량"], "온실가스 환산량")

        df = df.dropna(subset=["연도", "월"])
    except Exception as e:
        raise RuntimeError(f"[월별 집계 단계] 데이터 타입 정리 중 오류: {e}") from e

    try:
        if by_agency:
            g = df.groupby(["연도", "기관명", "월"], as_index=False)["온실가스 환산량"].sum()
            g = g.rename(columns={"온실가스 환산량": "월별 온실가스 환산량"})

            if include_total:
                total = df.groupby(["연도", "월"], as_index=False)["온실가스 환산량"].sum()
                total["기관명"] = "전체"
                total = total.rename(columns={"온실가스 환산량": "월별 온실가스 환산량"})
                g = pd.concat([g, total], ignore_index=True)

            return g.sort_values(["연도", "기관명", "월"]).reset_index(drop=True)

        g = df.groupby(["연도", "월"], as_index=False)["온실가스 환산량"].sum()
        g = g.rename(columns={"온실가스 환산량": "월별 온실가스 환산량"})
        return g.sort_values(["연도", "월"]).reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"[월별 집계 단계] groupby/집계 중 오류: {e}") from e


# 2) 연간 총 배출량 계산
def get_annual_ghg(
    df_std: pd.DataFrame,
    *,
    by_agency: bool = True,
    include_total: bool = True,
) -> pd.DataFrame:
    _ensure_not_empty(df_std, "표준 스키마")

    required = {"연도", "기관명", "온실가스 환산량"}
    missing = required - set(df_std.columns)
    if missing:
        raise ValueError(f"[연간 집계 단계] 필수 컬럼 누락: {missing}")

    try:
        df = df_std.copy()
        df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
        df["온실가스 환산량"] = _force_numeric(df["온실가스 환산량"], "온실가스 환산량")
        df = df.dropna(subset=["연도"])
    except Exception as e:
        raise RuntimeError(f"[연간 집계 단계] 데이터 타입 정리 중 오류: {e}") from e

    try:
        if by_agency:
            g = df.groupby(["연도", "기관명"], as_index=False)["온실가스 환산량"].sum()
            g = g.rename(columns={"온실가스 환산량": "연간 온실가스 배출량"})

            if include_total:
                total = df.groupby(["연도"], as_index=False)["온실가스 환산량"].sum()
                total["기관명"] = "전체"
                total = total.rename(columns={"온실가스 환산량": "연간 온실가스 배출량"})
                g = pd.concat([g, total], ignore_index=True)

            return g.sort_values(["연도", "기관명"]).reset_index(drop=True)

        g = df.groupby(["연도"], as_index=False)["온실가스 환산량"].sum()
        g = g.rename(columns={"온실가스 환산량": "연간 온실가스 배출량"})
        return g.sort_values(["연도"]).reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"[연간 집계 단계] groupby/집계 중 오류: {e}") from e


# 3) baseline 대비 감축률, 배출비율 계산
def calculate_reduction_metrics(
    annual_ghg_df: pd.DataFrame,
    baseline_map: Dict[int, float],
) -> pd.DataFrame:
    _ensure_not_empty(annual_ghg_df, "연간 배출량")

    if "연도" not in annual_ghg_df.columns or "연간 온실가스 배출량" not in annual_ghg_df.columns:
        raise ValueError(
            "[감축률 계산 단계] annual_ghg_df에는 '연도'와 '연간 온실가스 배출량' 컬럼이 필요합니다."
        )

    df = annual_ghg_df.copy()
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["연간 온실가스 배출량"] = _force_numeric(df["연간 온실가스 배출량"], "연간 온실가스 배출량")

    if not isinstance(baseline_map, dict):
        baseline_map = {}

    df["기준배출량"] = df["연도"].map(baseline_map)

    def _calc_ratio(row):
        baseline = row["기준배출량"]
        actual = row["연간 온실가스 배출량"]
        if baseline is None or pd.isna(baseline) or baseline == 0:
            return pd.NA
        return actual / baseline

    def _calc_reduction(row):
        baseline = row["기준배출량"]
        actual = row["연간 온실가스 배출량"]
        if baseline is None or pd.isna(baseline) or baseline == 0:
            return pd.NA
        return (baseline - actual) / baseline * 100.0

    try:
        df["배출비율"] = df.apply(_calc_ratio, axis=1)
        df["감축률(%)"] = df.apply(_calc_reduction, axis=1)
    except Exception as e:
        raise RuntimeError(f"[감축률 계산 단계] 계산 중 오류: {e}") from e

    return df


# 4) 최근 5개년 연간 배출량 리스트/DF
def get_recent_years_ghg(
    annual_ghg_df: pd.DataFrame,
    *,
    n_years: int = 5,
    base_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    _ensure_not_empty(annual_ghg_df, "연간 배출량")

    if "연도" not in annual_ghg_df.columns:
        raise ValueError("[최근 5개년 단계] annual_ghg_df에는 '연도' 컬럼이 필요합니다.")

    df = annual_ghg_df.copy()
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["연도"])

    if base_year is None:
        base_year = int(df["연도"].max())

    min_year = int(df["연도"].min())
    max_year = int(df["연도"].max())
    if base_year < min_year:
        base_year = max_year
    if base_year > max_year:
        base_year = max_year

    start_year = base_year - n_years + 1

    mask = (df["연도"] >= start_year) & (df["연도"] <= base_year)
    df_recent = df.loc[mask].copy()

    sort_cols = ["연도"]
    if "기관명" in df_recent.columns:
        sort_cols.append("기관명")

    df_recent = df_recent.sort_values(sort_cols).reset_index(drop=True)
    years = sorted(df_recent["연도"].dropna().astype(int).unique().tolist())

    return df_recent, years


# 5) 대시보드용 통합 헬퍼
def build_dashboard_datasets(
    df_std: pd.DataFrame,
    baseline_map: Dict[int, float],
) -> dict:
    """
    대시보드에서 바로 사용할 수 있는 집계 결과들을 한 번에 계산해 반환.
    """
    _ensure_not_empty(df_std, "표준 스키마")

    try:
        monthly_by_agency = get_monthly_ghg(df_std, by_agency=True, include_total=False)
    except Exception as e:
        raise RuntimeError(f"[대시보드 집계] 월별·기관별 집계 단계 오류: {e}") from e

    try:
        monthly_total = get_monthly_ghg(df_std, by_agency=False)
    except Exception as e:
        raise RuntimeError(f"[대시보드 집계] 월별(공단 전체) 집계 단계 오류: {e}") from e

    try:
        annual_by_agency = get_annual_ghg(df_std, by_agency=True, include_total=False)
    except Exception as e:
        raise RuntimeError(f"[대시보드 집계] 연간·기관별 집계 단계 오류: {e}") from e

    try:
        annual_total = get_annual_ghg(df_std, by_agency=False)
    except Exception as e:
        raise RuntimeError(f"[대시보드 집계] 연간(공단 전체) 집계 단계 오류: {e}") from e

    try:
        annual_total_with_baseline = calculate_reduction_metrics(
            annual_total,
            baseline_map=baseline_map,
        )
    except Exception:
        annual_total_with_baseline = annual_total.copy()
        annual_total_with_baseline["기준배출량"] = pd.NA
        annual_total_with_baseline["배출비율"] = pd.NA
        annual_total_with_baseline["감축률(%)"] = pd.NA

    recent5_total, recent5_years = get_recent_years_ghg(annual_total)

    return {
        "monthly_by_agency": monthly_by_agency,
        "monthly_total": monthly_total,
        "annual_by_agency": annual_by_agency,
        "annual_total": annual_total,
        "annual_total_with_baseline": annual_total_with_baseline,
        "recent5_total": recent5_total,
        "recent5_years": recent5_years,
    }


# 6) 전망분석 테이블
def build_projection_tables(
    annual_total: pd.DataFrame,
    annual_by_agency: pd.DataFrame,
    baseline_map: Dict[int, float],
    target_year: int,
) -> Dict[str, pd.DataFrame]:
    """
    [전망분석]용 테이블 생성 (공단 전체 + 소속기구별)
    """

    # 공단 전체
    row_total = annual_total[annual_total["연도"] == target_year]
    if row_total.empty:
        overall_df = pd.DataFrame(
            [
                {
                    "연도": target_year,
                    "기준배출량(tCO2eq)": pd.NA,
                    "이행연도 배출량(tCO2eq)": pd.NA,
                    "배출비율": pd.NA,
                    "감축률(%)": pd.NA,
                }
            ]
        )
    else:
        actual = float(row_total["연간 온실가스 배출량"].sum())
        baseline = float(baseline_map.get(target_year)) if target_year in baseline_map else pd.NA

        if baseline is pd.NA or pd.isna(baseline) or baseline == 0:
            ratio = pd.NA
            reduction = pd.NA
        else:
            ratio = actual / baseline
            reduction = (baseline - actual) / baseline * 100.0

        overall_df = pd.DataFrame(
            [
                {
                    "연도": target_year,
                    "기준배출량(tCO2eq)": baseline,
                    "이행연도 배출량(tCO2eq)": actual,
                    "배출비율": ratio,
                    "감축률(%)": reduction,
                }
            ]
        )

    # 소속기구별
    df_year_agency = annual_by_agency[annual_by_agency["연도"] == target_year].copy()
    if df_year_agency.empty:
        by_agency_df = pd.DataFrame(
            columns=[
                "기관명",
                "기준배출량(tCO2eq)",
                "이행연도 배출량(tCO2eq)",
                "배출비율",
                "감축률(%)",
            ]
        )
    else:
        df_year_agency["연간 온실가스 배출량"] = _force_numeric(
            df_year_agency["연간 온실가스 배출량"], "연간 온실가스 배출량"
        )

        total_actual = float(df_year_agency["연간 온실가스 배출량"].sum())
        baseline_total = baseline_map.get(target_year)

        if baseline_total is None or pd.isna(baseline_total) or total_actual == 0:
            df_year_agency["기준배출량(tCO2eq)"] = pd.NA
            df_year_agency["배출비율"] = pd.NA
            df_year_agency["감축률(%)"] = pd.NA
        else:
            baseline_total = float(baseline_total)
            share = df_year_agency["연간 온실가스 배출량"] / total_actual
            baseline_agency = baseline_total * share

            df_year_agency["기준배출량(tCO2eq)"] = baseline_agency

            ratio = df_year_agency["연간 온실가스 배출량"] / baseline_agency.replace(0, pd.NA)
            reduction = (baseline_agency - df_year_agency["연간 온실가스 배출량"]) / baseline_agency.replace(
                0, pd.NA
            ) * 100.0

            df_year_agency["배출비율"] = ratio
            df_year_agency["감축률(%)"] = reduction

        by_agency_df = df_year_agency[
            ["기관명", "기준배출량(tCO2eq)", "연간 온실가스 배출량", "배출비율", "감축률(%)"]
        ].rename(columns={"연간 온실가스 배출량": "이행연도 배출량(tCO2eq)"})

    return {"overall": overall_df, "by_agency": by_agency_df}


# 7) 피드백 테이블
def build_feedback_tables(
    annual_total: pd.DataFrame,
    annual_by_agency: pd.DataFrame,
    target_year: int,
) -> Dict[str, pd.DataFrame]:
    """
    [피드백]용 요약 테이블 생성 (공단 전체 + 소속기구별)
    """

    # 공단 전체
    df_total = annual_total.copy()
    df_total["연도"] = pd.to_numeric(df_total["연도"], errors="coerce").astype("Int64")
    df_total["연간 온실가스 배출량"] = _force_numeric(
        df_total["연간 온실가스 배출량"], "연간 온실가스 배출량"
    )

    row_cur = df_total[df_total["연도"] == target_year]
    row_prev = df_total[df_total["연도"] == target_year - 1]

    cur_val = float(row_cur["연간 온실가스 배출량"].sum()) if not row_cur.empty else pd.NA
    prev_val = float(row_prev["연간 온실가스 배출량"].sum()) if not row_prev.empty else pd.NA

    if prev_val is pd.NA or pd.isna(prev_val) or prev_val == 0:
        yoy_diff = pd.NA
        yoy_rate = pd.NA
    else:
        yoy_diff = cur_val - prev_val if not pd.isna(cur_val) else pd.NA
        yoy_rate = (cur_val - prev_val) / prev_val * 100.0 if not pd.isna(cur_val) else pd.NA

    window_years = df_total[df_total["연도"] <= target_year].sort_values("연도").tail(5)
    if len(window_years) >= 2:
        window_years["prev"] = window_years["연간 온실가스 배출량"].shift(1)
        valid = window_years["prev"] > 0
        yoy_series = (window_years["연간 온실가스 배출량"] - window_years["prev"]) / window_years["prev"] * 100.0
        recent5_avg = float(yoy_series[valid].mean()) if valid.any() else pd.NA
    else:
        recent5_avg = pd.NA

    feedback_overall = pd.DataFrame(
        [
            {
                "연도": target_year,
                "금년 배출량(tCO2eq)": cur_val,
                "전년 배출량(tCO2eq)": prev_val,
                "전년 대비 증감량(tCO2eq)": yoy_diff,
                "전년 대비 증감률(%)": yoy_rate,
                "최근5개년 평균 증감률(%)": recent5_avg,
            }
        ]
    )

    # 소속기구별
    df_agency = annual_by_agency.copy()
    df_agency["연도"] = pd.to_numeric(df_agency["연도"], errors="coerce").astype("Int64")
    df_agency["연간 온실가스 배출량"] = _force_numeric(
        df_agency["연간 온실가스 배출량"], "연간 온실가스 배출량"
    )

    cur_agency = df_agency[df_agency["연도"] == target_year][["기관명", "연간 온실가스 배출량"]].rename(
        columns={"연간 온실가스 배출량": "금년 배출량(tCO2eq)"}
    )
    prev_agency = df_agency[df_agency["연도"] == target_year - 1][
        ["기관명", "연간 온실가스 배출량"]
    ].rename(columns={"연간 온실가스 배출량": "전년 배출량(tCO2eq)"})

    merged = pd.merge(cur_agency, prev_agency, on="기관명", how="outer")

    merged["전년 대비 증감량(tCO2eq)"] = merged["금년 배출량(tCO2eq)"] - merged["전년 배출량(tCO2eq)"]
    merged["전년 대비 증감률(%)"] = (
        (merged["금년 배출량(tCO2eq)"] - merged["전년 배출량(tCO2eq)"])
        / merged["전년 배출량(tCO2eq)"]
        * 100.0
    )
    merged.loc[merged["전년 배출량(tCO2eq)"] <= 0, "전년 대비 증감률(%)"] = pd.NA

    start_year = target_year - 4
    window = df_agency[(df_agency["연도"] >= start_year) & (df_agency["연도"] <= target_year)]

    def _avg_yoy(group: pd.DataFrame) -> float:
        g = group.sort_values("연도").copy()
        g["prev"] = g["연간 온실가스 배출량"].shift(1)
        valid = g["prev"] > 0
        yoy = (g["연간 온실가스 배출량"] - g["prev"]) / g["prev"] * 100.0
        if valid.any():
            return float(yoy[valid].mean())
        return float("nan")

    recent5_agency = (
        window.groupby("기관명").apply(_avg_yoy).rename("최근5개년 평균 증감률(%)").reset_index()
    )

    feedback_agency = pd.merge(merged, recent5_agency, on="기관명", how="left")

    return {
        "overall": feedback_overall,
        "by_agency": feedback_agency[
            [
                "기관명",
                "금년 배출량(tCO2eq)",
                "전년 배출량(tCO2eq)",
                "전년 대비 증감량(tCO2eq)",
                "전년 대비 증감률(%)",
                "최근5개년 평균 증감률(%)",
            ]
        ],
    }
