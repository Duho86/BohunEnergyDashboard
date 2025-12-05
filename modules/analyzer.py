import json
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

SPEC_PATH = "master_energy_spec.json"


# ------------------------------------------------------------
# spec loader
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_spec() -> dict:
    if not os.path.exists(SPEC_PATH):
        st.error(f"사양 파일을 찾지 못했습니다: {SPEC_PATH}")
        return {}
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_meta():
    spec = load_spec()
    meta = spec.get("meta", {})
    ndc = meta.get("ndc_target_rate", 0.0417)
    analysis_years = meta.get("analysis_years")
    return ndc, analysis_years


def get_sheet1_org_order() -> List[str]:
    """시트1 컬럼 순서를 JSON에서 가져오기."""
    spec = load_spec()
    tmpl = spec.get("output_templates", {}).get("data_1_raw_analysis", {})
    layout = tmpl.get("layout", {})
    labels = layout.get("column_labels") or []
    return labels


# ============================================================
# 시트1: 백데이터 분석용 집계
# ============================================================

def build_sheet1_tables(year_to_raw: Dict[int, pd.DataFrame]):
    """
    시트1 백데이터 분석용 3개 표 생성:
      ① 연도 × 기관 에너지 사용량(U)
      ② 연도 × 기관 연면적
      ③ 연도별 3개년 평균 에너지 사용량(기관별)

    JSON spec 의 data_1_raw_analysis 레이아웃을 기준으로
    기관 순서 / 연도 순서를 정렬한다.
    """
    if not year_to_raw:
        return None, None, None

    years = sorted(year_to_raw.keys())

    # 기관 순서: JSON column_labels 우선, 없으면 df 에서 자동 추출
    org_order_spec = get_sheet1_org_order()
    if org_order_spec:
        org_order = org_order_spec.copy()
        # 혹시 데이터에 추가 기관이 있으면 뒤에 붙임
        for y in years:
            for name in year_to_raw[y]["기관명"].unique():
                if name not in org_order:
                    org_order.append(name)
    else:
        org_order = []
        for y in years:
            for name in year_to_raw[y]["기관명"].unique():
                if name not in org_order:
                    org_order.append(name)

    # -------------------------------
    # ① 연도 × 기관 에너지 사용량(U)
    # -------------------------------
    df_u = pd.DataFrame(index=org_order, columns=years, dtype="float64")

    for y in years:
        df = year_to_raw[y]
        s = df.groupby("기관명", as_index=True)["U"].sum()
        df_u[y] = s.reindex(org_order)

    df_u.loc["합계"] = df_u.sum(axis=0)
    df_u["합계"] = df_u.sum(axis=1)

    # -------------------------------
    # ② 연도 × 기관 연면적
    # -------------------------------
    df_area = pd.DataFrame(index=org_order, columns=years, dtype="float64")

    for y in years:
        df = year_to_raw[y]
        s = df.groupby("기관명", as_index=True)["연면적"].sum()
        df_area[y] = s.reindex(org_order)

    df_area.loc["합계"] = df_area.sum(axis=0)
    df_area["합계"] = df_area.sum(axis=1)

    # -------------------------------
    # ③ 3개년 평균 에너지 사용량 (기관별)
    # 각 연도별 직전 최대 3개년 평균
    # -------------------------------
    df_three = pd.DataFrame(index=org_order, columns=years, dtype="float64")

    for idx, y in enumerate(years):
        prev_years = years[max(0, idx - 3):idx]
        if not prev_years:
            df_three[y] = df_u[y]
        else:
            df_three[y] = df_u[prev_years].mean(axis=1)

    df_three.loc["합계"] = df_three.sum(axis=0)
    df_three["합계"] = df_three.sum(axis=1)

    return df_u, df_area, df_three


# ============================================================
# 시트2: 에너지 사용량 분석
# ============================================================

def compute_overall_sheet2(target_year: int, year_to_raw: Dict[int, pd.DataFrame]):
    """
    시트2 상단: 공단 전체 기준 표.

    출력 dict 은 다음 key 를 가진다.
      - 에너지사용량
      - 전년대비증감률
      - 3개년평균대비증감률
      - 의료시설평균W
      - 복지시설평균W
      - 기타시설평균W
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None

    years = sorted(year_to_raw.keys())
    idx = years.index(target_year)
    df = year_to_raw[target_year]

    total_u = df["U"].sum()

    # 전년 대비 증감률
    rate_yoy = None
    if idx > 0:
        prev_year = years[idx - 1]
        prev_total = year_to_raw[prev_year]["U"].sum()
        if prev_total != 0:
            rate_yoy = (total_u - prev_total) / prev_total

    # 3개년 평균 대비 증감률
    prev_years = years[max(0, idx - 3):idx]
    rate_vs_3yr = None
    if prev_years:
        prev_mean = sum(year_to_raw[y]["U"].sum() for y in prev_years) / len(prev_years)
        if prev_mean != 0:
            rate_vs_3yr = (total_u - prev_mean) / prev_mean

    # 시설구분별 평균 W
    w_med = w_wel = w_etc = None
    if "시설구분" in df.columns:
        g = df.groupby("시설구분")["W"].mean()
        w_med = g.get("의료시설")
        w_wel = g.get("복지시설")
        # 나머지는 기타시설로 집계
        etc_mask = ~g.index.isin(["의료시설", "복지시설"])
        if etc_mask.any():
            w_etc = g[etc_mask].mean()
    else:
        st.error("df_raw에 '시설구분' 컬럼이 없어 시설별 평균W를 계산할 수 없습니다.")

    return {
        "에너지사용량": total_u,
        "전년대비증감률": rate_yoy,
        "3개년평균대비증감률": rate_vs_3yr,
        "의료시설평균W": w_med,
        "복지시설평균W": w_wel,
        "기타시설평균W": w_etc,
    }


def compute_facility_sheet2(target_year: int, year_to_raw: Dict[int, pd.DataFrame]):
    """
    시트2 하단: 소속기구별 분석 표.

    JSON spec 의 org_level_current_year_metrics / data_2_usage_analysis.sections.by_org
    정의를 코드로 옮긴다.

    반환 DataFrame 컬럼:
      - 기관명
      - 시설구분
      - 연면적
      - 에너지 사용량(U)
      - 면적당 온실가스량(V)
      - 전체 대비 비중
      - 시설군 평균 대비 비율
      - 3개년 평균 대비 증감률
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None

    years = sorted(year_to_raw.keys())
    idx = years.index(target_year)
    df_target = year_to_raw[target_year].copy()

    if "시설구분" not in df_target.columns:
        st.error("df_raw에 '시설구분' 컬럼이 없어 시트2 하단 분석을 계산할 수 없습니다.")
        return None

    # 기관별 현재연도 사용량 / 면적 / 시설구분
    grp = df_target.groupby("기관명", as_index=True)
    cur_usage = grp["U"].sum()
    area = grp["연면적"].sum()
    facility_type = grp["시설구분"].agg(
        lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]
    )
    v_per_area = (grp["V"].sum() / area).replace([pd.NA, float("inf")], 0)

    total_u = cur_usage.sum()
    share = cur_usage / total_u if total_u != 0 else 0

    # 3개년 평균 U
    prev_years = years[max(0, idx - 3):idx]
    if prev_years:
        hist = {}
        for y in prev_years + [target_year]:
            df_y = year_to_raw[y]
            s = df_y.groupby("기관명")["U"].sum()
            hist[y] = s
        hist_df = pd.DataFrame(hist).fillna(0)
        avg_3yr = hist_df[prev_years].mean(axis=1)
        vs_3yr = (hist_df[target_year] - avg_3yr) / avg_3yr.replace(0, pd.NA)
    else:
        avg_3yr = pd.Series(index=cur_usage.index, dtype="float64")
        vs_3yr = pd.Series(index=cur_usage.index, dtype="float64")

    # 시설군별 평균 usage_per_area
    usage_per_area = cur_usage / area.replace(0, pd.NA)
    fac_group_avg = usage_per_area.groupby(facility_type).transform("mean")
    rel_to_group = usage_per_area / fac_group_avg.replace(0, pd.NA)

    out = pd.DataFrame(
        {
            "기관명": cur_usage.index,
            "시설구분": facility_type.values,
            "연면적": area.values,
            "에너지 사용량(U)": cur_usage.values,
            "면적당 온실가스량(V)": v_per_area.values,
            "전체 대비 비중": share.values,
            "시설군 평균 대비 비율": rel_to_group.values,
            "3개년 평균 대비 증감률": vs_3yr.values,
        }
    )

    # 기관명 순서는 JSON org_order 가 있으면 그 순서대로 정렬
    org_order = get_sheet1_org_order()
    if org_order:
        out["__order"] = out["기관명"].apply(
            lambda x: org_order.index(x) if x in org_order else len(org_order)
        )
        out = out.sort_values("__order").drop(columns="__order")

    return out


# ============================================================
# 시트3: 피드백
# ============================================================

def compute_overall_feedback(target_year: int, year_to_raw: Dict[int, pd.DataFrame]):
    """
    시트3 상단: 공단 전체 피드백 표.

    JSON rules.overall_recommended_usage_by_ndc 를 구현:
      recommended_usage = current_year_total_usage * (1 - ndc_rate)
      reduction_rate_yoy = -ndc_rate
      reduction_rate_vs_3yr = (recommended_usage - avg_3yr_usage) / avg_3yr_usage
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None

    ndc, _ = get_meta()
    years = sorted(year_to_raw.keys())
    idx = years.index(target_year)

    total_cur = year_to_raw[target_year]["U"].sum()

    # 3개년 평균
    prev_years = years[max(0, idx - 3):idx]
    avg_3yr = None
    if prev_years:
        avg_3yr = sum(year_to_raw[y]["U"].sum() for y in prev_years) / len(prev_years)

    recommended = total_cur * (1 - ndc)
    rate_yoy = -ndc
    rate_vs_3yr = None
    if avg_3yr and avg_3yr != 0:
        rate_vs_3yr = (recommended - avg_3yr) / avg_3yr

    return {
        "권장사용량": recommended,
        "전년대비감축률": rate_yoy,
        "3개년평균감축률": rate_vs_3yr,
    }


def compute_facility_feedback(target_year: int, year_to_raw: Dict[int, pd.DataFrame]):
    """
    시트3 하단: 소속기구별 피드백 2개 표.

    표1 컬럼:
      - 구분(기관명)
      - 사용 분포 순위
      - 3개년 평균 증가율 순위
      - 평균 에너지 사용량 순위
      - 권장 에너지 사용량
      - 권장 대비 비율

    표2 컬럼:
      - 구분(기관명)
      - 면적대비 과사용
      - 3개년 증가
      - 권장 초과
      - 에너지 사용량 관리대상
    """
    if target_year not in year_to_raw:
        st.error(f"{target_year}년 데이터가 존재하지 않습니다.")
        return None, None

    ndc, _ = get_meta()
    years = sorted(year_to_raw.keys())
    idx = years.index(target_year)
    df_t = year_to_raw[target_year].copy()

    # 기관별 집계
    grp = df_t.groupby("기관명", as_index=True)
    cur_usage = grp["U"].sum()
    area = grp["연면적"].sum()
    usage_per_area = cur_usage / area.replace(0, pd.NA)

    total_u = cur_usage.sum()
    usage_share = cur_usage / total_u if total_u != 0 else 0  # 사용 분포용

    # 3개년 평균 대비 증가율
    prev_years = years[max(0, idx - 3):idx]
    growth_rate = pd.Series(index=cur_usage.index, dtype="float64")
    if prev_years:
        hist = {}
        for y in prev_years:
            df_y = year_to_raw[y]
            s_y = df_y.groupby("기관명")["U"].sum()
            hist[y] = s_y
        hist_df = pd.DataFrame(hist).fillna(0)
        avg_3yr = hist_df.mean(axis=1)
        growth_rate = (cur_usage - avg_3yr) / avg_3yr.replace(0, pd.NA)

    # 권장 사용량 및 비율
    recommended = cur_usage * (1 - ndc)
    vs_recommended = cur_usage / recommended.replace(0, pd.NA)

    # 순위
    rank_usage = usage_share.rank(ascending=False, method="min")
    rank_growth = growth_rate.rank(ascending=False, method="min")
    rank_usage_per_area = usage_per_area.rank(ascending=False, method="min")

    table1 = pd.DataFrame(
        {
            "구분": cur_usage.index,
            "사용 분포 순위": rank_usage.values,
            "3개년 평균 증가율 순위": rank_growth.values,
            "평균 에너지 사용량 순위": rank_usage_per_area.values,
            "권장 에너지 사용량": recommended.values,
            "권장 대비 비율": vs_recommended.values,
        }
    )

    # 관리대상 플래그
    # 기준: 권장 대비 비율 > 1 또는 면적당 사용량 상위 N위
    N = 3
    topN_mask = rank_usage_per_area <= N
    excess_rec = vs_recommended > 1
    flag_area = ["O" if topN_mask.get(idx_) else "X" for idx_ in cur_usage.index]
    flag_growth = [
        "O" if (growth_rate.get(idx_) is not None and growth_rate.get(idx_) > 0) else "X"
        for idx_ in cur_usage.index
    ]
    flag_rec = ["O" if excess_rec.get(idx_) else "X" for idx_ in cur_usage.index]

    final_flag = []
    for i, org in enumerate(cur_usage.index):
        if flag_area[i] == "O" or flag_growth[i] == "O" or flag_rec[i] == "O":
            final_flag.append("O")
        else:
            final_flag.append("X")

    table2 = pd.DataFrame(
        {
            "구분": cur_usage.index,
            "면적대비 과사용": flag_area,
            "3개년 증가": flag_growth,
            "권장 초과": flag_rec,
            "에너지 사용량 관리대상": final_flag,
        }
    )

    return table1, table2


# ============================================================
# 서술형 코멘트 생성
# ============================================================

def generate_overall_comment(
    target_year: int, overall_metrics: dict, facility_table: pd.DataFrame
) -> str:
    """공단 전체 요약 코멘트."""
    total = overall_metrics.get("에너지사용량")
    yoy = overall_metrics.get("전년대비증감률")
    vs3 = overall_metrics.get("3개년평균대비증감률")

    # 상위 사용기관 3개
    top3 = (
        facility_table.sort_values("에너지 사용량(U)", ascending=False)
        .head(3)["기관명"]
        .tolist()
    )

    parts = [f"{target_year}년 공단 전체 에너지 사용량은 약 {total:,.0f}kWh 수준입니다."]
    if yoy is not None:
        parts.append(f"전년 대비로는 {yoy*100:.2f}% 변화가 나타났습니다.")
    if vs3 is not None:
        parts.append(f"최근 3개년 평균과 비교하면 {vs3*100:.2f}% 수준입니다.")
    if top3:
        parts.append(f"에너지 사용 비중이 큰 기관은 {', '.join(top3)} 순입니다.")

    return " ".join(parts)


def generate_org_comments(table1: pd.DataFrame, table2: pd.DataFrame) -> List[str]:
    """기관별 2~3문장 코멘트 리스트."""
    comments = []
    flags = table2.set_index("구분")

    for _, row in table1.iterrows():
        org = row["구분"]
        rank_usage = int(row["사용 분포 순위"])
        rank_growth = int(row["3개년 평균 증가율 순위"])
        rank_w = int(row["평균 에너지 사용량 순위"])
        vs_rec = row["권장 대비 비율"]

        flag_row = flags.loc[org]
        is_mgmt = flag_row["에너지 사용량 관리대상"] == "O"

        text = (
            f"{org}은(는) 사용 분포 {rank_usage}위, "
            f"3개년 평균 증가율 {rank_growth}위, "
            f"평균 에너지 사용량 {rank_w}위 수준입니다. "
        )
        text += f"권장 사용량 대비 현재 사용량은 약 {vs_rec*100:.1f}% 수준이며, "

        if is_mgmt:
            text += "여러 지표에서 과다 사용 또는 증가 경향이 나타나 관리 대상 기관으로 분류됩니다."
        else:
            text += "권장 수준 내에서 비교적 안정적으로 운영되고 있습니다."

        comments.append(text)

    return comments
