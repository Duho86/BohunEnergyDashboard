# app.py
# -*- coding: utf-8 -*-
"""
공단 에너지 사용량 · 온실가스 관리 대시보드 (최종 요구사항 반영 버전)

핵심 원칙
---------
- df_std(표준 스키마)는 사용하지 않음
- 모든 분석/피드백은 df_raw = loader.load_energy_raw_for_analysis(path) 기반
- 기준배출량 관련 기능/계산/텍스트 전면 제거
- 시트1/2/3 구조를 그대로 반영:
    * 시트1 → "에너지 사용량 파일 업로드" 탭의 백데이터 분석
    * 시트2 → "대시보드" 탭 내 "에너지 사용량 분석"
    * 시트3 → "대시보드" 탭 내 "피드백"
- 상단 기존 그래프 레이아웃(월별 추이, 연도별 추이)은 유지하되, 데이터 소스만 df_raw 기반 재집계
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from modules import loader
from modules.loader import EnergyDataError

# ------------------------------------------------------------
# 기본 경로 및 상수
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ENERGY_DIR = DATA_DIR / "energy"

NDC_RATE = 0.0417  # 온실가스 감축 목표 연평균 감축률 4.17%

# 기관 표시 순서 (요구서 기준)
FACILITY_ORDER = [
    "본사",
    "중앙병원",
    "부산병원",
    "광주병원",
    "대구병원",
    "대전병원",
    "인천병원",
    "교육연구원",
    "보훈원",
    "수원요양원",
    "광주요양원",
    "김해요양원",
    "대구요양원",
    "대전요양원",
    "남양주요양원",
    "원주요양원",
    "전주요양원",
    "재활체육센터",
    "휴양원",
]

# 원본 소속기관명 → 화면에 표시할 기관명 매핑 (예: 중앙보훈병원 → 중앙병원)
FACILITY_NAME_MAP = {
    "중앙보훈병원": "중앙병원",
    "부산보훈병원": "부산병원",
    "광주보훈병원": "광주병원",
    "대구보훈병원": "대구병원",
    "대전보훈병원": "대전병원",
    "인천보훈병원": "인천병원",
    "보훈교육연구원": "교육연구원",
    "보훈휴양원": "휴양원",
    "수원보훈요양원": "수원요양원",
    "광주보훈요양원": "광주요양원",
    "김해보훈요양원": "김해요양원",
    "대구보훈요양원": "대구요양원",
    "대전보훈요양원": "대전요양원",
    "남양주보훈요양원": "남양주요양원",
    "원주보훈요양원": "원주요양원",
    "전주보훈요양원": "전주요양원",
    "보훈재활체육센터": "재활체육센터",
}

# 시설구분(의료/복지/기타)
MEDICAL_FACILITIES = ["중앙병원", "부산병원", "광주병원", "대구병원", "대전병원", "인천병원"]
WELFARE_FACILITIES = [
    "수원요양원",
    "광주요양원",
    "김해요양원",
    "대구요양원",
    "대전요양원",
    "남양주요양원",
    "원주요양원",
    "전주요양원",
]
OTHER_FACILITIES = ["본사", "교육연구원", "보훈원", "재활체육센터", "휴양원"]


# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
def ensure_energy_dir() -> None:
    ENERGY_DIR.mkdir(parents=True, exist_ok=True)


def extract_year_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def normalize_facility_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if name in FACILITY_NAME_MAP:
        return FACILITY_NAME_MAP[name]
    return name


def get_facility_group(display_name: str) -> str:
    if display_name in MEDICAL_FACILITIES:
        return "의료시설"
    if display_name in WELFARE_FACILITIES:
        return "복지시설"
    if display_name in OTHER_FACILITIES:
        return "기타시설"
    return "기타시설"


# ------------------------------------------------------------
# df_raw 로딩 및 집계
# ------------------------------------------------------------
def load_all_raw_energy(base_dir: Path) -> Tuple[Dict[int, pd.DataFrame], List[str]]:
    """ENERGY_DIR 안의 연도별 엑셀을 df_raw(dict[연도])로 로딩"""
    ensure_energy_dir()
    year_to_df: Dict[int, pd.DataFrame] = {}
    issues: List[str] = []

    for xlsx_path in sorted(base_dir.glob("*.xlsx")):
        year = extract_year_from_filename(xlsx_path.name)
        if year is None:
            issues.append(f"연도 추출 실패: {xlsx_path.name}")
            continue
        try:
            df_raw = loader.load_energy_raw_for_analysis(xlsx_path)
            year_to_df[year] = df_raw
        except Exception as e:
            issues.append(f"{xlsx_path.name} 로딩 오류: {e}")

    return year_to_df, issues


def build_facility_metrics_for_year(year: int, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    단일 연도(df_raw)에 대해 기관별 U/V/W/연면적 집계.

    가정:
    - df_raw.columns[2]  : 소속기관명(C열)
    - df_raw.columns[20] : U열(에너지 사용량 합계)
    - df_raw.columns[21] : V열(면적당 온실가스 / 또는 V계열 지표)
    - df_raw.columns[22] : W열(연면적 기준 평균 에너지 사용량)
    - 연면적 컬럼: '연면적' 문자열 포함 컬럼 중 하나
    """
    cols = list(df_raw.columns)
    if len(cols) < 23:
        raise EnergyDataError("df_raw 컬럼 수가 예상보다 적어 U/V/W를 찾을 수 없습니다.")

    org_col = cols[2]
    U_col = cols[20]
    V_col = cols[21]
    W_col = cols[22]

    area_col = None
    for c in cols:
        if "연면적" in str(c):
            area_col = c
            break

    df = df_raw.copy()
    df[org_col] = df[org_col].astype(str).str.strip()

    # 숫자 전처리
    for c in [U_col, V_col, W_col] + ([area_col] if area_col else []):
        if c is None:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = []
    for raw_name, g in df.groupby(org_col):
        disp_name = normalize_facility_name(raw_name)
        if disp_name not in FACILITY_ORDER:
            # 사양서에 없는 기관은 일단 스킵 (필요하면 확장 가능)
            continue

        U = g[U_col].sum(skipna=True)
        V = g[V_col].sum(skipna=True)
        W = g[W_col].mean(skipna=True)  # 평균값 (행별 W가 이미 연면적 기준 지표라고 가정)
        area = None
        if area_col:
            # 연면적은 보통 한 행에만 들어가 있으므로 최대값 사용
            area = g[area_col].max(skipna=True)

        grouped.append(
            {
                "연도": year,
                "기관명": disp_name,
                "시설구분": get_facility_group(disp_name),
                "연면적": area,
                "U": U,
                "V": V,
                "W": W,
            }
        )

    df_fac = pd.DataFrame(grouped)

    # 기관 순서 정렬
    df_fac["기관명"] = pd.Categorical(df_fac["기관명"], categories=FACILITY_ORDER, ordered=True)
    df_fac = df_fac.sort_values(["기관명"]).reset_index(drop=True)
    return df_fac


def build_multi_year_facility_metrics(year_to_df_raw: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """여러 연도 df_raw → 기관별/연도별 메트릭 DataFrame"""
    frames = []
    for year, df_raw in year_to_df_raw.items():
        try:
            frames.append(build_facility_metrics_for_year(year, df_raw))
        except Exception:
            # 연도별 부분 오류는 개별로 무시하고 넘어감
            continue

    if not frames:
        return pd.DataFrame(columns=["연도", "기관명", "시설구분", "연면적", "U", "V", "W"])
    df_all = pd.concat(frames, ignore_index=True)
    return df_all


# ------------------------------------------------------------
# 시트1: 백데이터 분석용 집계
# ------------------------------------------------------------
def make_sheet1_energy_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """시트1 - 1. 에너지 사용량 (연도 x 기관)"""
    pivot = df_all.pivot_table(
        index="연도",
        columns="기관명",
        values="U",
        aggfunc="sum",
        fill_value=np.nan,
    ).reindex(columns=FACILITY_ORDER)
    pivot["합계"] = pivot.sum(axis=1, skipna=True)
    pivot = pivot.sort_index()
    return pivot


def make_sheet1_area_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """시트1 - 2. 연면적 (연도 x 기관)"""
    pivot = df_all.pivot_table(
        index="연도",
        columns="기관명",
        values="연면적",
        aggfunc="max",  # 연면적은 보통 연도 내에서 동일
    ).reindex(columns=FACILITY_ORDER)
    pivot["합계"] = pivot.sum(axis=1, skipna=True)
    pivot = pivot.sort_index()
    return pivot


def make_sheet1_3yr_avg_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    시트1 - 3. 연도별 에너지사용량의 3개년 평균 대비 분석용 "이전 1~3개년 평균 U" 테이블

    엑셀 로직에 최대한 맞춰:
    - 해당 연도 이전 연도들(최대 3개)의 U합계 평균
    """
    energy_table = make_sheet1_energy_table(df_all)
    years = sorted(energy_table.index.tolist())

    result = pd.DataFrame(index=years, columns=energy_table.columns, dtype=float)

    for i, y in enumerate(years):
        prev_years = years[max(0, i - 3) : i]  # y 이전 최대 3개 연도
        if not prev_years:
            # 과거 데이터가 없으면 해당 연도 U 값(엑셀 첫 행과 비슷한 역할)
            result.loc[y] = energy_table.loc[y]
        else:
            result.loc[y] = energy_table.loc[prev_years].mean(axis=0)

    return result


# ------------------------------------------------------------
# 시트2: 에너지 사용량 분석용 집계
# ------------------------------------------------------------
def compute_overall_sheet2(df_all: pd.DataFrame, year: int) -> Dict[str, float]:
    """
    시트2 상단(공단 전체 기준) 메트릭 계산:
    - 에너지 사용량(U 합계)
    - 면적당 온실가스 배출량(V 합계)
    - 3개년 평균 대비 증감률 (U 기준)
    - 의료/복지/기타 시설군 W평균
    """
    df_year = df_all[df_all["연도"] == year]
    if df_year.empty:
        return {}

    U_total = df_year["U"].sum(skipna=True)
    V_total = df_year["V"].sum(skipna=True)

    # 3개년 평균 U (이전 3개년 기준)
    past_years = [y for y in sorted(df_all["연도"].unique()) if y < year]
    past_years = past_years[-3:]
    if past_years:
        past_U = (
            df_all[df_all["연도"].isin(past_years)]
            .groupby("연도")["U"]
            .sum(skipna=True)
            .mean()
        )
        if past_U > 0:
            U_3yr_change = (U_total - past_U) / past_U * 100
        else:
            U_3yr_change = None
    else:
        U_3yr_change = None

    # 시설군별 W 평균
    def avg_W_for_group(names: List[str]) -> Optional[float]:
        sub = df_year[df_year["기관명"].isin(names)]
        if sub.empty:
            return None
        return sub["W"].mean(skipna=True)

    W_med = avg_W_for_group(MEDICAL_FACILITIES)
    W_wel = avg_W_for_group(WELFARE_FACILITIES)
    W_oth = avg_W_for_group(OTHER_FACILITIES)

    return {
        "U_total": U_total,
        "V_total": V_total,
        "U_3yr_change": U_3yr_change,
        "W_med": W_med,
        "W_wel": W_wel,
        "W_oth": W_oth,
    }


def compute_facility_sheet2(df_all: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    시트2 하단(소속기구별 분석) 표:
    - 구분
    - 시설구분
    - 연면적
    - 에너지 사용량(U)
    - 면적대비 에너지 사용비율 (여기서는 W를 면적대비 지표로 사용)
    - 에너지 사용 비중 (U 기관 / U 전체)
    - 3개년 평균 에너지 사용량 대비 증감률
    - 시설별 평균 면적 대비 에너지 사용비율 (W 기관 / 시설군 평균 W)
    """
    df_year = df_all[df_all["연도"] == year].copy()
    if df_year.empty:
        return pd.DataFrame()

    U_total = df_year["U"].sum(skipna=True)

    # 시설군별 W 평균
    group_W_mean = {}
    for gname, names in [
        ("의료시설", MEDICAL_FACILITIES),
        ("복지시설", WELFARE_FACILITIES),
        ("기타시설", OTHER_FACILITIES),
    ]:
        sub = df_year[df_year["기관명"].isin(names)]
        if sub.empty:
            group_W_mean[gname] = np.nan
        else:
            group_W_mean[gname] = sub["W"].mean(skipna=True)

    # 3개년 평균 대비 증감률
    years_all = sorted(df_all["연도"].unique())
    past_years_for_year = [y for y in years_all if y < year][-3:]

    def facility_3yr_change(row):
        name = row["기관명"]
        if not past_years_for_year:
            return np.nan
        past_vals = (
            df_all[(df_all["기관명"] == name) & (df_all["연도"].isin(past_years_for_year))]
            .groupby("연도")["U"]
            .sum(skipna=True)
        )
        if past_vals.empty:
            return np.nan
        past_avg = past_vals.mean()
        if past_avg == 0:
            return np.nan
        return (row["U"] - past_avg) / past_avg * 100

    # 기본 표
    df = df_year.copy()
    df["면적대비 에너지 사용비율"] = df["W"]  # W를 면적대비 에너지 사용비율로 해석
    df["에너지 사용 비중"] = df["U"] / U_total if U_total > 0 else np.nan
    df["3개년 평균 에너지 사용량 대비 증감률"] = df.apply(facility_3yr_change, axis=1)

    def avg_ratio(row):
        g = row["시설구분"]
        g_mean = group_W_mean.get(g)
        if pd.isna(row["W"]) or not g or pd.isna(g_mean) or g_mean == 0:
            return np.nan
        return row["W"] / g_mean

    df["시설별 평균 면적 대비 에너지 사용비율"] = df.apply(avg_ratio, axis=1)

    df_out = df[
        [
            "기관명",
            "시설구분",
            "연면적",
            "U",
            "V",
            "면적대비 에너지 사용비율",
            "에너지 사용 비중",
            "3개년 평균 에너지 사용량 대비 증감률",
            "시설별 평균 면적 대비 에너지 사용비율",
        ]
    ].rename(columns={"기관명": "구분", "U": "에너지 사용량", "V": "면적당 온실가스 배출량"})

    # 기관 순서 정렬
    df_out["구분"] = pd.Categorical(df_out["구분"], categories=FACILITY_ORDER, ordered=True)
    df_out = df_out.sort_values("구분").reset_index(drop=True)
    return df_out


# ------------------------------------------------------------
# 시트3: 피드백용 집계
# ------------------------------------------------------------
def compute_overall_feedback(df_all: pd.DataFrame, year: int) -> Dict[str, float]:
    """
    시트3 상단 (공단 전체 기준):
    - 권장 에너지 사용량: 전년 U합계 * (1 - NDC_RATE)
    - 전년대비 감축률: -NDC_RATE
    - 3개년 대비 감축률: (권장 - 이전 1~3개년 평균 U) / 그 평균
    """
    years_all = sorted(df_all["연도"].unique())
    if year not in years_all:
        return {}
    idx = years_all.index(year)
    if idx == 0:
        return {}  # 전년 없음

    prev_year = years_all[idx - 1]
    df_curr = df_all[df_all["연도"] == year]
    df_prev = df_all[df_all["연도"] == prev_year]

    U_prev = df_prev["U"].sum(skipna=True)
    if U_prev <= 0:
        return {}

    recommended_total = U_prev * (1 - NDC_RATE)

    # 3개년 평균(이전 1~3개년)
    prev_years_for_avg = years_all[max(0, idx - 3) : idx]
    df_prev3 = df_all[df_all["연도"].isin(prev_years_for_avg)]
    U_prev3_avg = df_prev3.groupby("연도")["U"].sum(skipna=True).mean()

    if U_prev3_avg and U_prev3_avg > 0:
        threeyr_rate = (recommended_total - U_prev3_avg) / U_prev3_avg
    else:
        threeyr_rate = None

    return {
        "prev_year": prev_year,
        "recommended_total": recommended_total,
        "prev_reduction_rate": -NDC_RATE,
        "three_year_reduction_rate": threeyr_rate,
    }


def compute_facility_feedback(df_all: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    시트3 소속기구별 피드백 표 + 관리대상 상세 표 생성.

    - 사용 분포 순위: U비중 기준 내림차순 rank (평균 순위 방식)
    - 에너지 3개년 평균 증가 순위: 기관별 3개년 평균 대비 증감률 기준 rank
    - 평균 에너지 사용량(연면적 기준) 순위: W 기준 rank
    - 권장 에너지 사용량: 전년 U * (1 - NDC_RATE)
    - 권장 사용량 대비 에너지 사용 비율: U현재 / 권장U
    - 에너지 사용량 관리 대상: 조건 기반 O/X
      (조건은 O/X 상세 표의 세 가지 플래그를 통합)
    """
    years_all = sorted(df_all["연도"].unique())
    if year not in years_all:
        return pd.DataFrame(), pd.DataFrame()
    idx = years_all.index(year)
    if idx == 0:
        # 첫 해는 전년 데이터가 없어 권장 사용량 산출이 애매하므로 빈 표
        return pd.DataFrame(), pd.DataFrame()
    prev_year = years_all[idx - 1]

    df_curr = df_all[df_all["연도"] == year].copy()
    df_prev = df_all[df_all["연도"] == prev_year].copy()

    # 기본 메트릭
    U_total = df_curr["U"].sum(skipna=True)

    # 기관별 현재/전년 U
    U_curr_by_fac = df_curr.set_index("기관명")["U"]
    U_prev_by_fac = df_prev.set_index("기관명")["U"]

    # 3개년 평균 증가율 계산
    past_years_for_avg = years_all[max(0, idx - 3) : idx]

    def facility_3yr_rate(name: str) -> float:
        past_vals = (
            df_all[(df_all["기관명"] == name) & (df_all["연도"].isin(past_years_for_avg))]
            .groupby("연도")["U"]
            .sum(skipna=True)
        )
        if past_vals.empty:
            return np.nan
        past_avg = past_vals.mean()
        if past_avg == 0:
            return np.nan
        curr = U_curr_by_fac.get(name, np.nan)
        if pd.isna(curr):
            return np.nan
        return (curr - past_avg) / past_avg

    records = []
    for _, row in df_curr.iterrows():
        name = row["기관명"]
        U_curr = row["U"]
        W_curr = row["W"]
        group = row["시설구분"]

        # 사용 비중
        share = U_curr / U_total if U_total > 0 else np.nan

        # 3개년 평균 증가율
        rate_3yr = facility_3yr_rate(name)

        # W기준 순위를 위해 일단 저장
        records.append(
            {
                "구분": name,
                "시설구분": group,
                "U_curr": U_curr,
                "W_curr": W_curr,
                "U_share": share,
                "rate_3yr": rate_3yr,
            }
        )

    df_fb = pd.DataFrame(records)
    if df_fb.empty:
        return pd.DataFrame(), pd.DataFrame()

    # W 그룹 평균
    group_W_mean = {}
    for gname, names in [
        ("의료시설", MEDICAL_FACILITIES),
        ("복지시설", WELFARE_FACILITIES),
        ("기타시설", OTHER_FACILITIES),
    ]:
        sub = df_fb[df_fb["시설구분"] == gname]
        if sub.empty:
            group_W_mean[gname] = np.nan
        else:
            group_W_mean[gname] = sub["W_curr"].mean(skipna=True)

    # 권장 에너지 사용량 (전년 U * (1 - NDC_RATE))
    recommended_by_fac = {}
    for name in df_fb["구분"].unique():
        prev_U = U_prev_by_fac.get(name, np.nan)
        if pd.isna(prev_U):
            recommended_by_fac[name] = np.nan
        else:
            recommended_by_fac[name] = prev_U * (1 - NDC_RATE)

    df_fb["권장 에너지 사용량"] = df_fb["구분"].map(recommended_by_fac)
    df_fb["권장 사용량 대비 에너지 사용 비율"] = df_fb["U_curr"] / df_fb["권장 에너지 사용량"]

    # 순위 계산 (엑셀처럼 값 기준 내림차순 rank, tie는 평균값)
    df_fb["사용 분포 순위"] = df_fb["U_share"].rank(ascending=False, method="average")
    df_fb["에너지 3개년 평균 증가 순위"] = df_fb["rate_3yr"].rank(
        ascending=False, method="average"
    )
    df_fb["평균 에너지 사용량(연면적 기준) 순위"] = df_fb["W_curr"].rank(
        ascending=False, method="average"
    )

    # 시설별 평균 면적 대비 에너지 사용비율(= 시트2의 '시설별 평균 면적 대비 에너지 사용비율'과 동일 로직)
    def w_ratio(row):
        g = row["시설구분"]
        g_mean = group_W_mean.get(g)
        if pd.isna(row["W_curr"]) or pd.isna(g_mean) or g_mean == 0:
            return np.nan
        return row["W_curr"] / g_mean

    df_fb["W_ratio_group"] = df_fb.apply(w_ratio, axis=1)

    # 관리 대상 상세 조건 (엑셀의 3개 플래그를 근사)
    # - 면적대비 에너지 과사용: W_ratio_group > 1.1
    # - 에너지 사용량 급증(3개년 평균대비): rate_3yr > 0.2 (20% 이상 증가)
    # - 권장량 대비 에너지 사용량 매우 초과: 권장 사용량 대비 비율 > 1.1
    def flag_area(row):
        return "O" if row["W_ratio_group"] > 1.1 else "X"

    def flag_rapid(row):
        return "O" if row["rate_3yr"] > 0.2 else "X"

    def flag_over(row):
        return "O" if row["권장 사용량 대비 에너지 사용 비율"] > 1.1 else "X"

    df_fb["면적대비 에너지 과사용"] = df_fb.apply(flag_area, axis=1)
    df_fb["에너지 사용량 급증(3개년 평균대비)"] = df_fb.apply(flag_rapid, axis=1)
    df_fb["권장량 대비 에너지 사용량 매우 초과"] = df_fb.apply(flag_over, axis=1)

    # 통합 관리 대상 (에너지 사용량 관리 대상): 세 조건 중 2개 이상 'O' 이면 'O'
    def overall_target(row):
        flags = [
            row["면적대비 에너지 과사용"],
            row["에너지 사용량 급증(3개년 평균대비)"],
            row["권장량 대비 에너지 사용량 매우 초과"],
        ]
        if flags.count("O") >= 2:
            return "O"
        return "X"

    df_fb["에너지 사용량 관리 대상"] = df_fb.apply(overall_target, axis=1)

    # 메인 피드백 표 (시트3 7~27행 구조)
    df_main = df_fb[
        [
            "구분",
            "사용 분포 순위",
            "에너지 3개년 평균 증가 순위",
            "평균 에너지 사용량(연면적 기준) 순위",
            "권장 에너지 사용량",
            "권장 사용량 대비 에너지 사용 비율",
            "에너지 사용량 관리 대상",
        ]
    ].copy()

    # 상세 플래그 표
    df_detail = df_fb[
        [
            "구분",
            "면적대비 에너지 과사용",
            "에너지 사용량 급증(3개년 평균대비)",
            "권장량 대비 에너지 사용량 매우 초과",
        ]
    ].copy()

    # 기관 순서 정렬
    for df_ in (df_main, df_detail):
        df_["구분"] = pd.Categorical(df_["구분"], categories=FACILITY_ORDER, ordered=True)
        df_.sort_values("구분", inplace=True)
        df_.reset_index(drop=True, inplace=True)

    return df_main, df_detail


# ------------------------------------------------------------
# AI 코멘트 생성
# ------------------------------------------------------------
def generate_overall_comment(
    df_all: pd.DataFrame,
    df_sheet2_fac: pd.DataFrame,
    df_fb_main: pd.DataFrame,
    year: int,
) -> str:
    """
    공단 전체 분석 코멘트 (보고서 형식)
    """
    df_year = df_all[df_all["연도"] == year]
    U_total = df_year["U"].sum(skipna=True)
    V_total = df_year["V"].sum(skipna=True)

    # 전년 대비 증감률
    years = sorted(df_all["연도"].unique())
    idx = years.index(year)
    if idx > 0:
        prev = years[idx - 1]
        U_prev = df_all[df_all["연도"] == prev]["U"].sum(skipna=True)
        if U_prev > 0:
            yoy = (U_total - U_prev) / U_prev * 100
        else:
            yoy = None
    else:
        yoy = None

    # 소속기구 중 U 증가율 상위 / 에너지 사용 비중 상위를 뽑아 간단히 언급
    top_share = df_sheet2_fac.sort_values("에너지 사용 비중", ascending=False).head(3)["구분"].tolist()
    top_3yr = (
        df_sheet2_fac.sort_values("3개년 평균 에너지 사용량 대비 증감률", ascending=False)
        .head(3)["구분"]
        .tolist()
    )

    comment_lines = []

    comment_lines.append(
        f"{year}년 기준 공단 전체 에너지 사용량(U 합계)은 약 {U_total:,.0f}로 집계되었으며, "
        f"온실가스 관련 지표(V 합계)는 약 {V_total:,.0f} 수준입니다."
    )
    if yoy is not None:
        direction = "증가" if yoy > 0 else "감소"
        comment_lines.append(
            f"전년 대비로는 약 {abs(yoy):.1f}% {direction}한 것으로, "
            "일부 기관의 사용량 변화가 전체 평균에 영향을 준 것으로 분석됩니다."
        )

    if top_share:
        comment_lines.append(
            f"에너지 사용 비중 측면에서는 "
            f"{', '.join(top_share)} 등이 전체 사용량에서 높은 비중을 차지하고 있습니다."
        )
    if top_3yr:
        comment_lines.append(
            f"최근 3개년 평균 대비 사용량 증가 폭이 큰 기관으로는 "
            f"{', '.join(top_3yr)} 등이 확인되며, 중점 관리가 필요합니다."
        )

    comment_lines.append(
        "향후 공단은 에너지 사용량이 크거나 증가율이 높은 기관을 중심으로 "
        "감축계획을 강화하고, 효율적으로 관리되고 있는 우수기관의 사례를 "
        "전 기관에 확산하는 방향으로 관리체계를 운영하는 것이 바람직합니다."
    )

    return "\n".join(comment_lines)


def generate_facility_comments(df_fb_main: pd.DataFrame) -> str:
    """
    기관별 피드백 코멘트 (테이블 하단)
    """
    lines = []
    for _, row in df_fb_main.iterrows():
        name = row["구분"]
        rank_share = row["사용 분포 순위"]
        rank_3yr = row["에너지 3개년 평균 증가 순위"]
        rank_W = row["평균 에너지 사용량(연면적 기준) 순위"]
        ratio = row["권장 사용량 대비 에너지 사용 비율"]
        target_flag = row["에너지 사용량 관리 대상"]

        if pd.isna(ratio):
            ratio_txt = "데이터 부족으로 권장 사용량 대비 분석이 곤란합니다."
        else:
            if ratio > 1.1:
                ratio_txt = f"권장 사용량 대비 약 { (ratio-1)*100:.1f}% 높은 수준입니다."
            elif ratio < 0.9:
                ratio_txt = f"권장 사용량 대비 약 {(1-ratio)*100:.1f}% 낮은 수준으로, 비교적 양호한 상태입니다."
            else:
                ratio_txt = "권장 사용량과 유사한 수준을 유지하고 있습니다."

        if target_flag == "O":
            need_txt = "에너지 사용량 관리 대상에 해당하며, 사용 증가 사유 분석 및 추가 감축 방안 검토가 필요합니다."
        else:
            need_txt = "현재 수준에서는 관리 대상 우선순위는 다소 낮으나, 지속적인 모니터링이 요구됩니다."

        lines.append(
            f"- **{name}**: 사용 비중 순위 {rank_share:.0f}위, "
            f"3개년 평균 증가 순위 {rank_3yr:.0f}위, "
            f"W기준 사용 수준 순위 {rank_W:.0f}위입니다. "
            f"{ratio_txt} {need_txt}"
        )

    return "\n".join(lines)


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="공단 에너지 사용량 · 온실가스 관리 대시보드",
    layout="wide",
)

st.title("공단 에너지 사용량 · 온실가스 관리 대시보드")

# 세션 상태
if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

# 상위 탭
tab_dashboard, tab_upload, tab_debug = st.tabs(
    ["📊 대시보드", "📂 에너지 사용량 파일 업로드", "🔧 디버그/진단"]
)

# ============================================================
# 📂 에너지 사용량 파일 업로드 탭 (시트1: 백데이터 분석)
# ============================================================
with tab_upload:
    st.header("에너지 사용량 파일 업로드")

    col_up1, col_up2 = st.columns([1.2, 2])

    with col_up1:
        uploaded_files = st.file_uploader(
            "에너지 사용량 관리 엑셀 파일(.xlsx) 업로드",
            type=["xlsx"],
            accept_multiple_files=True,
        )

        new_file_processed = False
        if uploaded_files:
            ensure_energy_dir()
            for f in uploaded_files:
                if f.name in st.session_state["processed_uploads"]:
                    st.info(f"{f.name} 은(는) 이미 업로드/저장되었습니다.")
                    continue
                try:
                    save_path = ENERGY_DIR / f.name
                    with open(save_path, "wb") as out:
                        out.write(f.getbuffer())
                    st.session_state["processed_uploads"].add(f.name)
                    st.success(f"{f.name} 저장 완료")
                    new_file_processed = True
                except Exception as e:
                    st.error(f"{f.name} 저장 실패: {e}")

        if new_file_processed:
            st.rerun()

    with col_up2:
        st.markdown("#### 저장된 에너지 사용량 파일 목록")
        ensure_energy_dir()
        files = sorted(ENERGY_DIR.glob("*.xlsx"))
        if not files:
            st.info("저장된 에너지 사용량 파일이 없습니다.")
        else:
            file_info = []
            for p in files:
                year = extract_year_from_filename(p.name)
                stat = p.stat()
                file_info.append(
                    {
                        "파일명": p.name,
                        "연도": year,
                        "크기(KB)": round(stat.st_size / 1024, 1),
                        "최종 수정": datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)

    st.markdown("---")

    # 백데이터 분석 (시트1 구조)
    year_to_raw, issues = load_all_raw_energy(ENERGY_DIR)
    if issues:
        with st.expander("⚠️ 로딩 중 발생한 이슈 확인"):
            for msg in issues:
                st.warning(msg)

    if not year_to_raw:
        st.info("백데이터 분석을 위해 최소 1개 이상의 에너지 사용량 파일이 필요합니다.")
    else:
        df_all = build_multi_year_facility_metrics(year_to_raw)
        if df_all.empty:
            st.info("df_raw 기반 기관별 집계를 생성할 수 없습니다.")
        else:
            st.subheader("1. 에너지 사용량 (시트1 구조)")

            tbl_energy = make_sheet1_energy_table(df_all)
            st.markdown("##### 1) 연도별 기관 에너지 사용량(U 합계)")
            st.dataframe(tbl_energy, use_container_width=True)

            st.subheader("2. 연면적 (시트1 구조)")
            tbl_area = make_sheet1_area_table(df_all)
            st.markdown("##### 2) 연도별 기관 연면적")
            st.dataframe(tbl_area, use_container_width=True)

            st.subheader("3. 연도별 에너지사용량의 3개년 평균 대비 분석 (시트1 구조)")
            tbl_3yr = make_sheet1_3yr_avg_table(df_all)
            st.markdown(
                "※ 각 연도별 에너지 사용량을 이전 1~3개년 평균과 비교하기 위한 기준값입니다."
            )
            st.dataframe(tbl_3yr, use_container_width=True)

# ============================================================
# 📊 대시보드 탭 (시트2 + 시트3)
# ============================================================
with tab_dashboard:
    st.header("대시보드")

    year_to_raw, issues_dash = load_all_raw_energy(ENERGY_DIR)
    if issues_dash:
        with st.expander("⚠️ 데이터 로딩 이슈"):
            for msg in issues_dash:
                st.warning(msg)

    if not year_to_raw:
        st.info("대시보드를 위해 에너지 사용량 파일을 먼저 업로드해 주세요.")
    else:
        df_all = build_multi_year_facility_metrics(year_to_raw)
        if df_all.empty:
            st.info("df_raw 기반 기관별 집계를 생성할 수 없습니다.")
        else:
            years_available = sorted(df_all["연도"].unique())
            default_year = max(years_available)
            selected_year = st.sidebar.selectbox(
                "분석 연도 선택",
                years_available,
                index=years_available.index(default_year),
            )

            df_curr_raw = year_to_raw.get(selected_year)
            if df_curr_raw is None:
                st.error(f"{selected_year}년 df_raw를 찾을 수 없습니다.")
            else:
                # ----------------------------------------
                # 상단: 기존 그래프 레이아웃 유지 (df_raw 기반)
                # ----------------------------------------
                st.markdown("## 에너지 사용량 추이")

                cols_graph = st.columns(2)

                # 월별 추이 그래프 (df_raw의 월별 컬럼 합계)
                with cols_graph[0]:
                    st.markdown("#### 월별 에너지 사용량 추이")

                    # 월 컬럼 탐지 (예: '1월' ~ '12월')
                    month_cols = [
                        c
                        for c in df_curr_raw.columns
                        if isinstance(c, str)
                        and c.endswith("월")
                        and c[0].isdigit()
                    ]
                    df_month_chart = pd.DataFrame()
                    if month_cols:
                        df_tmp = df_curr_raw.copy()
                        for c in month_cols:
                            df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce")
                        month_sum = df_tmp[month_cols].sum(axis=0, skipna=True)
                        df_month_chart = pd.DataFrame(
                            {"월": month_cols, "에너지 사용량": month_sum.values}
                        )
                        df_month_chart.set_index("월", inplace=True)
                        st.line_chart(df_month_chart)
                    else:
                        st.info("월별 에너지 사용량 컬럼(1월~12월)을 찾지 못했습니다.")

                # 최근 연도별 추이 그래프 (U합계 기준 5개년)
                with cols_graph[1]:
                    st.markdown("#### 연도별 에너지 사용량 추이 (최대 5개년)")

                    df_year_total = (
                        df_all.groupby("연도")["U"].sum(skipna=True).reset_index()
                    )
                    df_year_total = df_year_total.sort_values("연도").tail(5)
                    df_year_total = df_year_total.set_index("연도")
                    st.bar_chart(df_year_total)

                st.markdown("---")

                # ----------------------------------------
                # 하단: 에너지 사용량 분석 / 피드백 (시트2/3)
                # ----------------------------------------
                subtab_analysis, subtab_feedback = st.tabs(["에너지 사용량 분석", "피드백"])

                # ========================================
                # 시트2: 에너지 사용량 분석
                # ========================================
                with subtab_analysis:
                    st.subheader("에너지 사용량 분석 (시트2)")

                    # 공단 전체 기준 (상단 블록)
                    overall = compute_overall_sheet2(df_all, selected_year)
                    if not overall:
                        st.info("선택 연도에 대해 공단 전체 분석 값을 계산할 수 없습니다.")
                    else:
                        st.markdown("### 1. 공단 전체 기준")

                        k1, k2, k3 = st.columns(3)
                        k1.metric(
                            "에너지 사용량(U 합계)",
                            f"{overall['U_total']:,.0f}",
                        )
                        k2.metric(
                            "면적당 온실가스 관련 지표(V 합계)",
                            f"{overall['V_total']:,.0f}",
                        )
                        if overall["U_3yr_change"] is None:
                            k3.metric("3개년 평균 에너지 사용량 대비 증감률", "-")
                        else:
                            k3.metric(
                                "3개년 평균 에너지 사용량 대비 증감률",
                                f"{overall['U_3yr_change']:.1f}%",
                            )

                        g1, g2, g3 = st.columns(3)
                        for col, label, key in [
                            (g1, "의료시설 평균(W)", "W_med"),
                            (g2, "복지시설 평균(W)", "W_wel"),
                            (g3, "기타시설 평균(W)", "W_oth"),
                        ]:
                            val = overall.get(key)
                            if val is None or pd.isna(val):
                                col.metric(label, "-")
                            else:
                                col.metric(label, f"{val:.3f}")

                    # 소속기구별 분석 표
                    st.markdown("### 2. 소속기구별 에너지 사용량 분석")

                    df_sheet2_fac = compute_facility_sheet2(df_all, selected_year)
                    if df_sheet2_fac.empty:
                        st.info("소속기구별 분석 표를 생성할 수 없습니다.")
                    else:
                        st.dataframe(df_sheet2_fac, use_container_width=True)

                # ========================================
                # 시트3: 피드백
                # ========================================
                with subtab_feedback:
                    st.subheader("피드백 (시트3)")

                    # 공단 전체 기준 피드백
                    st.markdown("### 1. 공단 전체 기준")

                    overall_fb = compute_overall_feedback(df_all, selected_year)
                    if not overall_fb:
                        st.info("공단 전체 피드백을 계산하기 위한 전년/과거 데이터가 부족합니다.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "권장 에너지 사용량",
                            f"{overall_fb['recommended_total']:,.0f}",
                            help=f"{overall_fb['prev_year']}년 U합계에 NDC {NDC_RATE*100:.2f}% 감축률을 적용한 값",
                        )
                        c2.metric(
                            "전년대비 감축률",
                            f"{overall_fb['prev_reduction_rate']*100:.2f}%",
                        )
                        if overall_fb["three_year_reduction_rate"] is None:
                            c3.metric("3개년 대비 감축률", "-")
                        else:
                            c3.metric(
                                "3개년 대비 감축률",
                                f"{overall_fb['three_year_reduction_rate']*100:.2f}%",
                            )

                    st.markdown("---")

                    # 소속기구별 피드백 표 + 상세 표
                    st.markdown("### 2. 소속기구별 피드백")

                    df_fb_main, df_fb_detail = compute_facility_feedback(
                        df_all, selected_year
                    )
                    if df_fb_main.empty:
                        st.info("소속기구별 피드백을 계산하기 위한 전년/과거 데이터가 부족합니다.")
                    else:
                        st.markdown("#### (1) 소속기구별 피드백 표")
                        st.dataframe(df_fb_main, use_container_width=True)

                        st.markdown("#### (2) 에너지 사용량 관리 대상 상세")
                        st.dataframe(df_fb_detail, use_container_width=True)

                        st.markdown("---")
                        st.markdown("### 3. 최종 피드백 문장")

                        # 전체 코멘트
                        overall_comment = generate_overall_comment(
                            df_all, df_sheet2_fac, df_fb_main, selected_year
                        )
                        st.markdown("#### (1) 공단 전체 분석 코멘트")
                        st.write(overall_comment)

                        # 기관별 코멘트
                        st.markdown("#### (2) 소속기구별 분석 코멘트")
                        facility_comment = generate_facility_comments(df_fb_main)
                        st.markdown(facility_comment)

# ============================================================
# 🔧 디버그 / 진단 탭
# ============================================================
with tab_debug:
    st.header("디버그 / 구조 진단")

    st.markdown("### 1. 엑셀 구조 진단")
    uploaded_debug_file = st.file_uploader(
        "엑셀 구조 진단용 파일 업로드 (.xlsx)", type=["xlsx"], key="debug_uploader"
    )
    if uploaded_debug_file:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_debug_file.read())
            tmp_path = Path(tmp.name)

        try:
            res = loader.validate_excel_file(tmp_path)
            st.json(res)
        except Exception as e:
            st.error(f"구조 진단 실패: {e}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    st.markdown("---")
    st.markdown("### 2. df_raw 미리보기")

    ensure_energy_dir()
    files = sorted(ENERGY_DIR.glob("*.xlsx"))
    if not files:
        st.info("저장된 에너지 사용량 파일이 없습니다.")
    else:
        debug_file = st.selectbox(
            "미리보기할 파일 선택", [p.name for p in files], key="debug_file_select"
        )
        if debug_file:
            path = ENERGY_DIR / debug_file
            try:
                df_raw_dbg = loader.load_energy_raw_for_analysis(path)
                st.write(f"df_raw shape: {df_raw_dbg.shape}")
                st.dataframe(df_raw_dbg.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"df_raw 로딩 실패: {e}")

    st.markdown("---")
    st.markdown("### 3. loader 모듈 정보")

    try:
        import inspect

        st.code(loader.__file__, language="text")
        st.write(dir(loader))
        st.code(inspect.getsource(loader), language="python")
    except Exception as e:
        st.error(f"loader 소스 확인 실패: {e}")
