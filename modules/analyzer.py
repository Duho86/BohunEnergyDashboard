# modules/analyzer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd


# ============================================================
# 내부 유틸
# ============================================================

def _ensure_not_empty(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        raise ValueError(f"{name} 데이터가 비어 있습니다. 먼저 에너지 사용량 파일을 업로드해 주세요.")


def _force_numeric(s: pd.Series, col_name: str) -> pd.Series:
    """집계에 사용하는 컬럼을 무조건 float 로 강제 변환."""
    try:
        s_num = pd.to_numeric(s, errors="coerce")
        return s_num.astype("float64")
    except Exception as e:
        raise RuntimeError(f"[숫자 변환] '{col_name}' 컬럼을 숫자로 변환하는 중 오류: {e}") from e


# ============================================================
# 1) 월별 · 기관별 온실가스 환산량 집계
# ============================================================

def get_monthly_ghg(
    df_std: pd.DataFrame, *, by_agency: bool = True, include_total: bool = False
) -> pd.DataFrame:
    """표준 스키마(df_std)를 월별 온실가스 환산량 집계 형태로 변환한다."""
    _ensure_not_empty(df_std, "표준 스키마")

    required = {"연도", "기관명", "월", "온실가스 환산량"}
    missing = required - set(df_std.columns)
    if missing:
        raise ValueError(f"[월별 집계 단계] 필수 컬럼 누락: {missing}")

    df = df_std.copy()

    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["월"] = pd.to_numeric(df["월"], errors="coerce").astype("Int64")
    df["온실가스 환산량"] = _force_numeric(df["온실가스 환산량"], "온실가스 환산량")

    df = df.dropna(subset=["연도", "월"])

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


# ============================================================
# 2) 연간 온실가스 배출량 집계
# ============================================================

def get_annual_ghg(
    df_std: pd.DataFrame, *, by_agency: bool = True, include_total: bool = True
) -> pd.DataFrame:
    """표준 스키마(df_std)를 연간 온실가스 배출량 집계 형태로 변환한다."""
    _ensure_not_empty(df_std, "표준 스키마")

    required = {"연도", "기관명", "온실가스 환산량"}
    missing = required - set(df_std.columns)
    if missing:
        raise ValueError(f"[연간 집계 단계] 필수 컬럼 누락: {missing}")

    df = df_std.copy()

    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["온실가스 환산량"] = _force_numeric(df["온실가스 환산량"], "온실가스 환산량")
    df = df.dropna(subset=["연도"])

    if by_agency:
        g = df.groupby(["연도", "기관명"], as_index=False)["온실가스 환산량"].sum()
        g = g.rename(columns={"온실가스 환산량": "연간 온실가스 배출량"})

        if include_total:
            tot = df.groupby("연도", as_index=False)["온실가스 환산량"].sum()
            tot["기관명"] = "전체"
            tot = tot.rename(columns={"온실가스 환산량": "연간 온실가스 배출량"})
            g = pd.concat([g, tot], ignore_index=True)

        return g.sort_values(["연도", "기관명"]).reset_index(drop=True)

    g = df.groupby("연도", as_index=False)["온실가스 환산량"].sum()
    g = g.rename(columns={"온실가스 환산량": "연간 온실가스 배출량"})
    return g.sort_values("연도").reset_index(drop=True)


# ============================================================
# 3) 기준배출량 관련 유틸 (현재 앱에서는 사용 안 함)
#    - 과거 호환성을 위해 함수만 남겨 둠
# ============================================================

def calculate_reduction_metrics(
    annual_ghg_df: pd.DataFrame, baseline_map: Dict[int, float]
) -> pd.DataFrame:
    """기준배출량 대비 배출비율/감축률 계산 (현재 앱에서는 미사용)."""
    _ensure_not_empty(annual_ghg_df, "연간 배출량")

    df = annual_ghg_df.copy()
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["연간 온실가스 배출량"] = _force_numeric(df["연간 온실가스 배출량"], "연간 온실가스 배출량")

    df["기준배출량"] = df["연도"].map(baseline_map)

    def r_ratio(row):
        b = row["기준배출량"]
        a = row["연간 온실가스 배출량"]
        if b is None or pd.isna(b) or b == 0:
            return pd.NA
        return a / b

    def r_reduction(row):
        b = row["기준배출량"]
        a = row["연간 온실가스 배출량"]
        if b is None or pd.isna(b) or b == 0:
            return pd.NA
        return (b - a) / b * 100

    df["배출비율"] = df.apply(r_ratio, axis=1)
    df["감축률(%)"] = df.apply(r_reduction, axis=1)

    return df


# ============================================================
# 4) 최근 5개년 연간 배출량 조회
# ============================================================

def get_recent_years_ghg(
    annual_ghg_df: pd.DataFrame, *, n_years: int = 5, base_year: Optional[int] = None
):
    _ensure_not_empty(annual_ghg_df, "연간 배출량")

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

    df_recent = df_recent.sort_values("연도").reset_index(drop=True)
    years = sorted(df_recent["연도"].dropna().astype(int).unique().tolist())

    return df_recent, years


# ============================================================
# 5) 대시보드용 집계 데이터 패키지
# ============================================================

def build_dashboard_datasets(df_std: pd.DataFrame) -> dict:
    """대시보드 상단 그래프/지표용 집계 데이터 패키지 생성.
    (기준배출량과 무관하게 온실가스 환산량 기준 집계만 수행)
    """
    _ensure_not_empty(df_std, "표준 스키마")

    monthly_by_agency = get_monthly_ghg(df_std, by_agency=True)
    monthly_total = get_monthly_ghg(df_std, by_agency=False)

    annual_by_agency = get_annual_ghg(df_std, by_agency=True)
    annual_total = get_annual_ghg(df_std, by_agency=False)

    return {
        "monthly_by_agency": monthly_by_agency,
        "monthly_total": monthly_total,
        "annual_by_agency": annual_by_agency,
        "annual_total": annual_total,
    }


# ============================================================
# (옛) 전망분석/피드백 테이블 더미 구현
# ============================================================

def build_projection_tables(*args, **kwargs):
    return {"overall": pd.DataFrame(), "by_agency": pd.DataFrame()}


def build_feedback_tables(*args, **kwargs):
    return {"overall": pd.DataFrame(), "by_agency": pd.DataFrame()}
