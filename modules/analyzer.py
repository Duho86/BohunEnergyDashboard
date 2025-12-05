import pandas as pd
import numpy as np
import streamlit as st


# ================================================================
# 공통 유틸
# ================================================================
def three_year_avg(series):
    """
    3개년 평균: 이전 1~3개년 평균 사용.
    1개년만 있으면 1개년 평균.
    2개년이면 2개년 평균.
    """
    valid = series.dropna()
    if len(valid) == 0:
        return None
    return valid.mean()


# ================================================================
# 시트 1 — 백데이터 분석
# ================================================================
def build_sheet1_tables(year_to_raw: dict):
    """
    시트1: 연도×기관 에너지사용량(U), 연면적, 3개년 평균 대비 분석
    PDF 원본과 동일한 표 구조 반환
    """

    if len(year_to_raw) == 0:
        return None, None, None

    # 기관명 전체 목록(연도별 기관 일치 가정)
    sample_year = list(year_to_raw.keys())[0]
    org_list = year_to_raw[sample_year]["기관명"].tolist()

    # -------------------------------
    # ① 연도 x 기관 에너지 사용량(U)
    # -------------------------------
    df_u = pd.DataFrame(index=org_list)

    for y, df in year_to_raw.items():
        df_u[y] = df["에너지사용량"].values

    df_u["합계"] = df_u.sum(axis=1)
    df_u.loc["합계"] = df_u.sum(axis=0)

    # -------------------------------
    # ② 연도 x 기관 연면적
    # -------------------------------
    df_area = pd.DataFrame(index=org_list)

    for y, df in year_to_raw.items():
        df_area[y] = df["연면적"].values

    df_area["합계"] = df_area.sum(axis=1)
    df_area.loc["합계"] = df_area.sum(axis=0)

    # -------------------------------
    # ③ 3개년 평균 대비 분석
    # -------------------------------
    df_three = pd.DataFrame(index=org_list)

    for y in sorted(year_to_raw.keys()):
        prev_years = [yy for yy in year_to_raw.keys() if yy < y]
        prev_years = prev_years[-3:]  # 최근 3개년

        colname = y
        df_three[colname] = np.nan

        if len(prev_years) == 0:
            # 최초 반영 연도: 실적 그대로
            df_three[colname] = year_to_raw[y]["에너지사용량"].values
        else:
            prev_values = df_u[prev_years].mean(axis=1)
            df_three[colname] = prev_values.values

    df_three["합계"] = df_three.sum(axis=1)
    df_three.loc["합계"] = df_three.sum(axis=0)

    return df_u, df_area, df_three


# ================================================================
# 시트 2 — 에너지 사용량 분석
# ================================================================
def compute_overall_sheet2(target_year: int, year_to_raw: dict):
    """
    시트2 상단 공단 전체 분석
    PDF 값과 동일 계산
    """
    if target_year not in year_to_raw:
        return None

    df = year_to_raw[target_year]

    total_u = df["에너지사용량"].sum()

    # 전년 대비 증감률
    prev_year = target_year - 1
    if prev_year in year_to_raw:
        prev_u = year_to_raw[prev_year]["에너지사용량"].sum()
        rate_prev = (total_u - prev_u) / prev_u if prev_u != 0 else np.nan
    else:
        rate_prev = np.nan

    # 3개년 평균 대비 증감률
    prev_years = [y for y in year_to_raw.keys() if y < target_year]
    prev_years = prev_years[-3:]

    if len(prev_years) > 0:
        avg_prev = np.mean([year_to_raw[y]["에너지사용량"].sum() for y in prev_years])
        rate_three = (total_u - avg_prev) / avg_prev if avg_prev != 0 else np.nan
    else:
        rate_three = np.nan

    # 시설구분 평균(W)
    facility_groups = df.groupby("시설구분")
    w_avg = facility_groups["면적대비사용비율"].mean().to_dict()

    return {
        "에너지사용량": total_u,
        "전년대비증감": rate_prev,
        "3개년평균대비증감": rate_three,
        "시설구분평균": w_avg  # {"의료시설": 0.61, ...}
    }


def compute_facility_sheet2(target_year: int, year_to_raw: dict):
    """
    시트2 하단 기관별 분석 표
    PDF와 동일한 열 구성
    """
    if target_year not in year_to_raw:
        return None

    df = year_to_raw[target_year].copy()

    total_u = df["에너지사용량"].sum()

    # 전년들 3개년 평균
    prev_years = [y for y in year_to_raw.keys() if y < target_year]
    prev_years = prev_years[-3:]

    if len(prev_years) > 0:
        prev_u = pd.concat([year_to_raw[y]["에너지사용량"] for y in prev_years], axis=1).mean(axis=1)
    else:
        prev_u = pd.Series([np.nan]*len(df))

    df["면적대비에너지비율"] = df["에너지사용량"] / df["연면적"]
    df["에너지비중"] = df["에너지사용량"] / total_u
    df["3개년평균대비증감률"] = (df["에너지사용량"] - prev_u.values) / prev_u.values

    # 시설군 평균 V 대비(= W)
    group_v = df.groupby("시설구분")["면적대비에너지비율"].mean().to_dict()
    df["시설군평균대비비율"] = df.apply(
        lambda row: row["면적대비에너지비율"] / group_v[row["시설구분"]]
        if row["시설구분"] in group_v else np.nan,
        axis=1
    )

    return df


# ================================================================
# 시트 3 — 피드백 (권장 사용량, 순위, O/X)
# ================================================================
def compute_overall_feedback(target_year: int, year_to_raw: dict):
    """
    시트3 상단 공단 전체 피드백
    PDF 기준: NDC = 4.17%
    """
    if target_year not in year_to_raw:
        return None

    df = year_to_raw[target_year]
    total_u = df["에너지사용량"].sum()

    # NDC 4.17% 기준
    # 권장 사용량 = 전년 사용량 × (1 - 0.0417)
    prev_year = target_year - 1
    if prev_year in year_to_raw:
        prev_u = year_to_raw[prev_year]["에너지사용량"].sum()
        recommended = prev_u * (1 - 0.0417)
    else:
        recommended = total_u  # 전년도 없음 → 실적 기준치

    # 전년대비 감축률
    rate_prev = (recommended - total_u) / total_u

    # 3개년 평균 대비 감축률
    prev_years = [y for y in year_to_raw.keys() if y < target_year]
    prev_years = prev_years[-3:]

    if len(prev_years) > 0:
        avg_prev = np.mean([year_to_raw[y]["에너지사용량"].sum() for y in prev_years])
        rate_three = (recommended - avg_prev) / avg_prev
    else:
        rate_three = np.nan

    return {
        "권장사용량": recommended,
        "전년대비감축률": rate_prev,
        "3개년평균감축률": rate_three
    }


def compute_facility_feedback(target_year: int, year_to_raw: dict):
    """
    시트3 하단 두 개 표:
    ① 기관별 피드백 요약
    ② 관리대상 상세(O/X)
    PDF의 기준 그대로 적용
    """

    if target_year not in year_to_raw:
        return None, None

    df = year_to_raw[target_year].copy()

    # -----------------------
    # 권장 사용량 (기관별)
    # -----------------------
    prev_year = target_year - 1
    if prev_year in year_to_raw:
        prev_df = year_to_raw[prev_year]
        recommended_each = prev_df["에너지사용량"] * (1 - 0.0417)
    else:
        recommended_each = df["에너지사용량"]

    total_u = df["에너지사용량"].sum()

    # -----------------------
    # 3개항목 순위
    # -----------------------
    # 사용 분포 순위(에너지비중)
    df["에너지비중"] = df["에너지사용량"] / total_u
    df["사용분포순위"] = df["에너지비중"].rank(ascending=False).astype(int)

    # 3개년 평균 증가 순위
    prev_years = [y for y in year_to_raw.keys() if y < target_year]
    prev_years = prev_years[-3:]
    if len(prev_years) > 0:
        prev_u_mean = pd.concat(
            [year_to_raw[y]["에너지사용량"] for y in prev_years],
            axis=1
        ).mean(axis=1)
        increase_rate = (df["에너지사용량"] - prev_u_mean) / prev_u_mean
    else:
        increase_rate = pd.Series([0]*len(df))

    df["증가순위"] = increase_rate.rank(ascending=False).astype(int)

    # 연면적 대비 평균(W) 기준 순위
    df["면적대비에너지"] = df["에너지사용량"] / df["연면적"]
    df["평균에너지순위"] = df["면적대비에너지"].rank(ascending=False).astype(int)

    # -----------------------
    # 권장 사용량 대비 비율
    # -----------------------
    df["권장사용량"] = recommended_each.values
    df["권장대비비율"] = df["에너지사용량"] / df["권장사용량"]

    # -----------------------
    # 관리대상 O/X (3개 플래그)
    # -----------------------
    # PDF 기준에 따라 그대로 적용:
    # ① 면적대비 에너지 과사용(O/X)
    #    기준 = 시설군 평균 대비 > 1.1 (PDF 실제값 분석 기반)
    facility_group_avg = df.groupby("시설구분")["면적대비에너지"].mean().to_dict()
    df["과사용"] = df.apply(
        lambda row: "O" if (row["면적대비에너지"] /
                            facility_group_avg[row["시설구분"]]) > 1.1 else "X",
        axis=1
    )

    # ② 에너지 급증 여부 = 3개년 평균 대비 증가율 > 20%
    df["급증"] = df.apply(
        lambda row: "O" if row["에너지사용량"] > row["권장사용량"] * 1.2 else "X",
        axis=1
    )

    # ③ 권장량 대비 매우 초과 = 권장대비비율 > 1.5
    df["권장초과"] = df["권장대비비율"].apply(lambda x: "O" if x > 1.5 else "X")

    # 최종 관리 대상 = 3개 중 하나라도 O
    df["관리대상"] = df.apply(
        lambda row: "O" if ("O" in [row["과사용"], row["급증"], row["권장초과"]]) else "X",
        axis=1
    )

    # -----------------------
    # 최종 출력용 데이터프레임 2개
    # -----------------------
    df_feedback1 = df[[
        "기관명", "사용분포순위", "증가순위", "평균에너지순위",
        "권장사용량", "권장대비비율", "관리대상"
    ]]

    df_feedback2 = df[[
        "기관명", "과사용", "급증", "권장초과"
    ]]

    return df_feedback1, df_feedback2
