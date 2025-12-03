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
        # 이 단계에서 다시 한 번 에러를 감싸서 어느 컬럼인지 알려줌
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

        # 연도/월/온실가스 모두 숫자 캐스팅
        df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
        df["월"] = pd.to_numeric(df["월"], errors="coerce").astype("Int64")
        df["온실가스 환산량"] = _force_numeric(df["온실가스 환산량"], "온실가스 환산량")

        # 연도/월 정보가 없는 행 제거
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
        # baseline 관련 문제는 대시보드 전체를 죽이지 않고 메시지로만 안내할 수 있도록
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
