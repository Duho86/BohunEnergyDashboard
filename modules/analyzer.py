import pandas as pd
import streamlit as st


# ============================================================
# 시트1: 백데이터 분석용 집계
# ============================================================

def build_sheet1_tables(year_to_raw: dict[int, pd.DataFrame]):
    """
    시트1 백데이터 분석용 3개 표 생성:
      ① 연도 × 기관 에너지 사용량(U)
      ② 연도 × 기관 연면적
      ③ 연도별 3개년 평균 에너지 사용량(기관별)

    - df_raw의 표준 컬럼: 기관명, U, 연면적 을 사용한다.
    - 기관 순서는 최초 등장 순서를 기준으로 union.
      (정상 데이터라면 연도별 기관 목록이 동일하므로 예시 엑셀과 동일한 순서가 됨)
    """
    if not year_to_raw:
        return None, None, None

    years = sorted(year_to_raw.keys())

    # 기관 순서 결정 (union + 최초 등장 순서 유지)
    org_order: list[str] = []
    for y in years:
        df = year_to_raw[y]
        for name in df["기관명"].tolist():
            if name not in org_order:
                org_order.append(name)

    # -------------------------------
    # ① 연도 × 기관 에너지 사용량(U)
    #   - 기관명이 중복될 수 있으므로 반드시 groupby 후 집계
    # -------------------------------
    df_u = pd.DataFrame(index=org_order)

    for y in years:
        df = year_to_raw[y]
        # 기관별 에너지 사용량 합계 (중복 제거)
        s = df.groupby("기관명", as_index=True)["U"].sum()
        # 기관 순서에 맞춰 재정렬
        s = s.reindex(org_order)
        df_u[y] = s

    df_u["합계"] = df_u.sum(axis=1)
    df_u.loc["합계"] = df_u.sum(axis=0)

    # -------------------------------
    # ② 연도 × 기관 연면적
    #   - 마찬가지로 기관별 집계 후 사용
    # -------------------------------
    df_area = pd.DataFrame(index=org_order)

    for y in years:
        df = year_to_raw[y]
        # 기관별 연면적 합계 (중복 제거)
        s = df.groupby("기관명", as_index=True)["연면적"].sum()
        s = s.reindex(org_order)
        df_area[y] = s

    df_area["합계"] = df_area.sum(axis=1)
    df_area.loc["합계"] = df_area.sum(axis=0)

    # -------------------------------
    # ③ 3개년 평균 에너지 사용량 (기관별)
    #    - 각 연도별로 직전 최대 3개년 U의 평균
    # -------------------------------
    df_three = pd.DataFrame(index=org_order)

    for idx, y in enumerate(years):
        prev_years = years[max(0, idx - 3):idx]
        if not prev_years:
            # 최초 연도는 해당 연도 U 그대로 (예시 엑셀 기준)
            df_three[y] = df_u[y]
        else:
            prev_mean = df_u[prev_years].mean(axis=1)
            df_three[y] = prev_mean

    df_three["합계"] = df_three.sum(axis=1)
    df_three.loc["합계"] = df_three.sum(axis=0)

    return df_u, df_area, df_three


# ============================================================
# 시트2: 에너지 사용량 분석
# ============================================================

def compute_overall_sheet2(target_year: int, year_to_raw: dict[int, pd.DataFrame]):
    """
    시트2 상단: 공단 전체 기준 표용 집계.

    - 에너지 사용량(U 합계)
    - 전체 면적당 온실가스 배출량(V)
      (시설별 V * 연면적 합 / 전체 연면적)
    - 3개년 평균 에너지 사용량 대비 증감률
    - 시설구분별 평균 에너지 사용량(W 평균)
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None

    years = sorted(year_to_raw.keys())
    target_idx = years.index(target_year)

    df = year_to_raw[target_year]

    total_u = df["U"].sum()
    total_area = df["연면적"].sum()

    # 전체 면적당 온실가스 배출량(V): 면적 가중 평균
    total_v = (df["V"] * df["연면적"]).sum() / total_area if total_area != 0 else 0.0

    # 직전연도 대비 증감률
    if target_idx == 0:
        rate_prev = None
    else:
        prev_year = years[target_idx - 1]
        prev_total_u = year_to_raw[prev_year]["U"].sum()
        rate_prev = (total_u - prev_total_u) / prev_total_u if prev_total_u != 0 else None

    # 3개년 평균 대비 증감률
    prev_years = years[max(0, target_idx - 3):target_idx]
    if not prev_years:
        rate_three = None
    else:
        prev_mean_u = sum(year_to_raw[y]["U"].sum() for y in prev_years) / len(prev_years)
        rate_three = (total_u - prev_mean_u) / prev_mean_u if prev_mean_u != 0 else None

    # 시설구분별 평균 W
    if "시설구분" not in df.columns:
        st.error("df_raw에 '시설구분' 컬럼이 없어 시트2 상단 시설구분별 평균을 계산할 수 없습니다.")
        return None

    facility_groups = df.groupby("시설구분")
    w_avg_by_group = facility_groups["W"].mean().to_dict()

    return {
        "에너지사용량": total_u,
        "전체면적당온실가스": total_v,
        "전년대비증감": rate_prev,
        "3개년평균대비증감": rate_three,
        "시설구분평균W": w_avg_by_group,
    }


def compute_facility_sheet2(target_year: int, year_to_raw: dict[int, pd.DataFrame]):
    """
    시트2 하단: 소속기구별 분석 표.

    열 구성(예시 엑셀 시트2 7~행 기준):
      - 구분(기관명)
      - 시설구분
      - 에너지 사용량(U)
      - 면적당 온실가스 배출량(V)
      - 공단 에너지 사용량 분포 비율
      - 평균 에너지 사용량(연면적 기준) 대비 사용비율
      - 3개년 평균 에너지 사용량 대비 증감률
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None

    years = sorted(year_to_raw.keys())
    target_idx = years.index(target_year)
    df = year_to_raw[target_year].copy()

    if "시설구분" not in df.columns:
        st.error("df_raw에 '시설구분' 컬럼이 없어 시트2 하단 분석을 계산할 수 없습니다.")
        return None

    # 기본 값
    total_u = df["U"].sum()

    # 공단 에너지 사용량 분포 비율
    df["공단에너지분포비율"] = df["U"] / total_u if total_u != 0 else 0

    # 시설군 평균 W 대비 비율
    w_group_mean = df.groupby("시설구분")["W"].mean().to_dict()
    df["시설군평균W"] = df["시설구분"].map(w_group_mean)
    df["평균에너지사용비율"] = df["W"] / df["시설군평균W"]

    # 3개년 평균 대비 증감률
    prev_years = years[max(0, target_idx - 3):target_idx]
    if prev_years:
        # 기관별 U 이력 집계 (중복 제거 후 사용)
        history = {}
        for y in prev_years:
            df_y = year_to_raw[y]
            s_y = df_y.groupby("기관명", as_index=True)["U"].sum()
            history[y] = s_y

        hist_df = pd.DataFrame(history)
        three_mean = hist_df.mean(axis=1)

        df = df.set_index("기관명")
        df["3개년평균U"] = three_mean
        df["3개년평균U"] = df["3개년평균U"].fillna(0)
        df["3개년평균대비증감률"] = df.apply(
            lambda row: (row["U"] - row["3개년평균U"]) / row["3개년평균U"]
            if row["3개년평균U"] != 0
            else None,
            axis=1,
        )
        df = df.reset_index()
    else:
        df["3개년평균U"] = None
        df["3개년평균대비증감률"] = None

    # 출력용 열 구성
    out = df[
        [
            "기관명",
            "시설구분",
            "U",
            "V",
            "공단에너지분포비율",
            "평균에너지사용비율",
            "3개년평균대비증감률",
        ]
    ].copy()

    out = out.rename(
        columns={
            "기관명": "구분",
            "U": "에너지사용량(U)",
            "V": "면적당온실가스배출량(V)",
            "공단에너지분포비율": "공단에너지사용분포비율",
            "평균에너지사용비율": "평균에너지사용량대비사용비율",
        }
    )

    return out


# ============================================================
# 시트3: 피드백 (공단 전체 + 소속기구별)
# ============================================================

# NDC / 권장 사용량 설정값
# 👉 실제 예시 엑셀 시트3에서 사용하는 값과 반드시 대조해서 맞춰야 함
NDC_RATE = 0.0417  # 4.17%


def compute_overall_feedback(target_year: int, year_to_raw: dict[int, pd.DataFrame]):
    """
    시트3 상단: 공단 전체 피드백용 값 계산.
      - 권장 에너지 사용량
      - 전년대비 감축률 (NDC 기반)
      - 3개년 평균 대비 감축률
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None

    years = sorted(year_to_raw.keys())
    target_idx = years.index(target_year)

    df = year_to_raw[target_year]
    total_u = df["U"].sum()

    # 직전연도 기준 권장사용량 = 직전연도 U * (1 - NDC_RATE)
    if target_idx == 0:
        # 직전연도 없으면 권장사용량 = 현재 사용량
        recommended = total_u
        rate_prev = None
    else:
        prev_year = years[target_idx - 1]
        prev_total_u = year_to_raw[prev_year]["U"].sum()
        recommended = prev_total_u * (1 - NDC_RATE)
        rate_prev = -NDC_RATE  # NDC 기준 감축률

    # 3개년 평균 대비 감축률
    prev_years = years[max(0, target_idx - 3):target_idx]
    if not prev_years:
        rate_three = None
    else:
        three_mean = sum(year_to_raw[y]["U"].sum() for y in prev_years) / len(prev_years)
        rate_three = (recommended - three_mean) / three_mean if three_mean != 0 else None

    return {
        "권장사용량": recommended,
        "전년대비감축률": rate_prev,
        "3개년평균감축률": rate_three,
    }


def compute_facility_feedback(target_year: int, year_to_raw: dict[int, pd.DataFrame]):
    """
    시트3 하단: 소속기구별 피드백 2개 표 생성.

    첫 번째 표(기관별 피드백 요약) 예시 열:
      - 구분(기관명)
      - 사용 분포 순위
      - 3개년 평균 증가 순위
      - 평균 에너지 사용량 순위
      - 권장 에너지 사용량
      - 권장 사용량 대비 비율

    두 번째 표(관리대상 O/X 상세) 예시 열:
      - 구분(기관명)
      - 면적대비 에너지 과사용 여부
      - 3개년 평균 증가 여부
      - 권장 사용량 대비 과다 여부
      - 종합 관리대상 (O/X)

    ⚠ 구체적인 조건/임계값은 반드시 예시 엑셀 시트3 수식을 확인해 맞춰야 한다.
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None, None

    years = sorted(year_to_raw.keys())
    target_idx = years.index(target_year)
    df = year_to_raw[target_year].copy()

    # 기본 지표
    total_u = df["U"].sum()
    df["사용분포"] = df["U"] / total_u if total_u != 0 else 0

    # 3개년 평균 U (기관별)
    prev_years = years[max(0, target_idx - 3):target_idx]
    if prev_years:
        history = {}
        for y in prev_years:
            df_y = year_to_raw[y]
            s_y = df_y.groupby("기관명", as_index=True)["U"].sum()
            history[y] = s_y

        hist_df = pd.DataFrame(history)
        df = df.set_index("기관명")
        df["3개년평균U"] = hist_df.mean(axis=1)
        df["3개년평균U"] = df["3개년평균U"].fillna(0)
        df["3개년증가율"] = df.apply(
            lambda row: (row["U"] - row["3개년평균U"]) / row["3개년평균U"]
            if row["3개년평균U"] != 0
            else None,
            axis=1,
        )
        df = df.reset_index()
    else:
        df["3개년평균U"] = None
        df["3개년증가율"] = None

    # 시설군 평균 W
    w_group_mean = df.groupby("시설구분")["W"].mean().to_dict()
    df["시설군평균W"] = df["시설구분"].map(w_group_mean)
    df["W비율"] = df["W"] / df["시설군평균W"]

    # 권장 사용량 (기관별) = 직전연도 기관별 U * (1 - NDC_RATE)
    if target_idx == 0:
        df["권장사용량"] = df["U"]
    else:
        prev_year = years[target_idx - 1]
        df_prev_raw = year_to_raw[prev_year]
        df_prev = df_prev_raw.groupby("기관명", as_index=True)["U"].sum()

        df = df.set_index("기관명")
        df["직전연도U"] = df_prev
        df["권장사용량"] = df["직전연도U"] * (1 - NDC_RATE)
        df = df.reset_index()

    df["권장대비비율"] = df.apply(
        lambda row: row["U"] / row["권장사용량"]
        if row["권장사용량"] not in (0, None)
        else None,
        axis=1,
    )

    # ---- 표1: 순위/비율 요약 ----
    df_rank = df.copy()

    # 순위: 값이 클수록 높은 사용/증가 → 1위
    df_rank["사용분포순위"] = df_rank["사용분포"].rank(ascending=False, method="min")
    df_rank["증가율순위"] = df_rank["3개년증가율"].rank(ascending=False, method="min")
    df_rank["W순위"] = df_rank["W"].rank(ascending=False, method="min")

    table1 = df_rank[
        [
            "기관명",
            "사용분포순위",
            "증가율순위",
            "W순위",
            "권장사용량",
            "권장대비비율",
        ]
    ].rename(
        columns={
            "기관명": "구분",
            "사용분포순위": "사용분포순위",
            "증가율순위": "3개년평균증가순위",
            "W순위": "평균에너지사용량순위",
        }
    )

    # ---- 표2: 관리대상 O/X 플래그 ----
    # ⚠ 아래 임계값은 "예시 엑셀 시트3의 실제 기준"과 맞춰 조정해야 함
    W_EXCESS_THRESHOLD = 1.0      # 예: 시설군 평균 대비 W비율 > 1.0 이면 과사용
    INCREASE_THRESHOLD = 0.0      # 예: 3개년 평균 대비 증가(>0) 시 위험
    RECOMM_EXCESS_THRESHOLD = 1.0  # 예: 권장사용량 이상(>=1.0) 이면 과다

    df_flag = df.copy()
    df_flag["면적대비과사용"] = df_flag["W비율"].apply(
        lambda v: "O" if v is not None and v > W_EXCESS_THRESHOLD else "X"
    )
    df_flag["3개년증가"] = df_flag["3개년증가율"].apply(
        lambda v: "O" if v is not None and v > INCREASE_THRESHOLD else "X"
    )
    df_flag["권장초과"] = df_flag["권장대비비율"].apply(
        lambda v: "O" if v is not None and v > RECOMM_EXCESS_THRESHOLD else "X"
    )

    def _agg_flag(row):
        flags = [row["면적대비과사용"], row["3개년증가"], row["권장초과"]]
        return "O" if any(f == "O" for f in flags) else "X"

    df_flag["에너지사용량관리대상"] = df_flag.apply(_agg_flag, axis=1)

    table2 = df_flag[
        [
            "기관명",
            "면적대비과사용",
            "3개년증가",
            "권장초과",
            "에너지사용량관리대상",
        ]
    ].rename(columns={"기관명": "구분"})

    return table1, table2
