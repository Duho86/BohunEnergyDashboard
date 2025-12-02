# modules/feedback.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import pandas as pd


def _analyze_trend_recent_years(
    recent_df: Optional[pd.DataFrame],
    current_year: int,
) -> str:
    """
    최근 연도 에너지 사용량(온실가스 배출량) 추세를 분석해 한국어 문장으로 요약.
    """
    if recent_df is None or recent_df.empty:
        return "최근 연도별 배출량 추세를 분석할 수 있는 데이터가 부족합니다."

    df = recent_df.copy()
    if "연도" not in df.columns or "연간 온실가스 배출량" not in df.columns:
        return "최근 연도별 배출량 추세를 분석할 수 있는 데이터가 부족합니다."

    df = df.sort_values("연도").reset_index(drop=True)

    cur = df[df["연도"] == current_year]
    if cur.empty:
        start_year = int(df["연도"].iloc[0])
        end_year = int(df["연도"].iloc[-1])
        first_val = df["연간 온실가스 배출량"].iloc[0]
        last_val = df["연간 온실가스 배출량"].iloc[-1]
        diff = last_val - first_val
        diff_pct = diff / first_val * 100 if first_val else 0

        if diff > 0:
            return (
                f"{start_year}년부터 {end_year}년까지 연간 온실가스 배출량은 "
                f"약 {diff_pct:,.1f}% 증가하는 추세를 보이고 있습니다."
            )
        elif diff < 0:
            return (
                f"{start_year}년부터 {end_year}년까지 연간 온실가스 배출량은 "
                f"약 {abs(diff_pct):,.1f}% 감소하는 추세를 보이고 있습니다."
            )
        else:
            return (
                f"{start_year}년부터 {end_year}년까지 연간 온실가스 배출량은 "
                f"큰 변화 없이 유사한 수준을 유지하고 있습니다."
            )

    cur_val = float(cur["연간 온실가스 배출량"].iloc[0])
    cur_year = int(cur["연도"].iloc[0])

    prev_year = cur_year - 1
    prev = df[df["연도"] == prev_year]

    max_row = df.loc[df["연간 온실가스 배출량"].idxmax()]
    min_row = df.loc[df["연간 온실가스 배출량"].idxmin()]

    sentences = []

    if not prev.empty:
        prev_val = float(prev["연간 온실가스 배출량"].iloc[0])
        diff = cur_val - prev_val
        diff_pct = diff / prev_val * 100 if prev_val else 0

        if diff > 0:
            sentences.append(
                f"{cur_year}년 연간 배출량은 전년({prev_year}년) 대비 "
                f"약 {diff_pct:,.1f}%({diff:,.0f} tCO2eq) 증가하였습니다."
            )
        elif diff < 0:
            sentences.append(
                f"{cur_year}년 연간 배출량은 전년({prev_year}년) 대비 "
                f"약 {abs(diff_pct):,.1f}%({abs(diff):,.0f} tCO2eq) 감소하였습니다."
            )
        else:
            sentences.append(
                f"{cur_year}년 연간 배출량은 전년({prev_year}년)과 거의 동일한 수준입니다."
            )
    else:
        sentences.append(
            f"{cur_year}년 연간 배출량은 {cur_val:,.0f} tCO2eq로 집계되었으며, "
            "전년 데이터는 존재하지 않습니다."
        )

    max_year = int(max_row["연도"])
    max_val = float(max_row["연간 온실가스 배출량"])
    min_year = int(min_row["연도"])
    min_val = float(min_row["연간 온실가스 배출량"])

    if cur_year == max_year:
        sentences.append(
            f"또한, 최근 분석 기간 중 {cur_year}년이 가장 높은 배출량을 기록하고 있어 "
            "추가적인 감축 노력이 필요한 시점입니다."
        )
    elif cur_year == min_year:
        sentences.append(
            f"또한, 최근 분석 기간 중 {cur_year}년이 가장 낮은 배출량을 기록하고 있어 "
            "감축 정책의 효과가 일부 나타나고 있는 것으로 해석됩니다."
        )
    else:
        if cur_val > (max_val + min_val) / 2:
            sentences.append(
                f"{cur_year}년 배출 수준은 최근 연도들 중 상위권에 해당하며, "
                "추가 감축 여력이 남아 있는 것으로 판단됩니다."
            )
        else:
            sentences.append(
                f"{cur_year}년 배출 수준은 최근 연도들 중 중간 이하 수준으로, "
                "현재의 감축 기조를 유지하는 것이 중요합니다."
            )

    return " ".join(sentences)


def generate_overall_feedback(
    year: int,
    actual_emission: Optional[float],
    baseline_emission: Optional[float],
    reduction_rate_pct: Optional[float],
    ratio_to_baseline: Optional[float],
    recent_total_df: Optional[pd.DataFrame] = None,
    current_month: Optional[int] = None,
) -> str:
    """
    공단 전체 이행연도 배출량 정보 + 기준배출량 + 감축률 + 최근 추세를
    한국어 보고서 스타일 문단으로 생성.
    """
    paragraphs = []

    # 1. 기본 현황
    if actual_emission is None or pd.isna(actual_emission):
        paragraphs.append(
            f"{year}년 온실가스 배출량은 아직 집계되지 않았거나, "
            "데이터가 충분하지 않아 정확한 분석이 어렵습니다."
        )
        trend_text = _analyze_trend_recent_years(recent_total_df, year)
        paragraphs.append(trend_text)
        return "\n\n".join(paragraphs)

    if current_month is not None:
        paragraphs.append(
            f"{year}년 {current_month}월 기준 공단 전체 온실가스 배출량은 "
            f"총 {actual_emission:,.0f} tCO2eq로 집계되었습니다."
        )
    else:
        paragraphs.append(
            f"{year}년 공단 전체 온실가스 배출량은 "
            f"총 {actual_emission:,.0f} tCO2eq로 집계되었습니다."
        )

    # 2. 기준배출량 대비 분석
    if baseline_emission is None or pd.isna(baseline_emission) or baseline_emission == 0:
        paragraphs.append(
            "해당 연도에 대한 기준배출량 정보가 설정되어 있지 않아 "
            "감축률과 목표달성 수준을 평가하기는 어렵습니다. "
            "baseline.json에 기준배출량을 등록하면, 목표 대비 이행 수준을 정량적으로 비교할 수 있습니다."
        )
    else:
        if ratio_to_baseline is not None and not pd.isna(ratio_to_baseline):
            ratio_pct = ratio_to_baseline * 100
            base_sentence = (
                f"기준배출량은 {baseline_emission:,.0f} tCO2eq이며, "
                f"{year}년 배출량은 기준의 약 {ratio_pct:,.1f}% 수준입니다."
            )
        else:
            base_sentence = (
                f"기준배출량은 {baseline_emission:,.0f} tCO2eq이며, "
                f"{year}년 배출량과의 정량 비교가 가능합니다."
            )

        if reduction_rate_pct is None or pd.isna(reduction_rate_pct):
            extra = ""
        else:
            if reduction_rate_pct > 0:
                extra = (
                    f" 기준배출량 대비 약 {reduction_rate_pct:,.1f}%를 감축한 것으로, "
                    "감축 목표 이행이 일부 진행되고 있는 것으로 판단됩니다."
                )
            elif reduction_rate_pct < 0:
                extra = (
                    f" 기준배출량을 약 {abs(reduction_rate_pct):,.1f}% 초과 배출한 상태로, "
                    "추가적인 감축 계획 수립과 관리 강화가 필요한 수준입니다."
                )
            else:
                extra = (
                    " 기준배출량과 동일한 수준으로 배출하고 있어, "
                    "향후 감축을 위한 추가 노력이 요구됩니다."
                )

        paragraphs.append(base_sentence + extra)

        diff = baseline_emission - actual_emission
        if diff > 0:
            paragraphs.append(
                f"현재 배출량은 기준배출량보다 약 {diff:,.0f} tCO2eq 낮은 수준으로, "
                "당초 설정된 감축 목표를 초과 달성하고 있습니다. "
                "다만, 향후에도 동일한 수준의 감축기조를 유지하는 것이 중요합니다."
            )
        elif diff < 0:
            need = abs(diff)
            paragraphs.append(
                f"기준배출량을 기준으로 볼 때 약 {need:,.0f} tCO2eq의 추가 감축이 필요한 상황입니다. "
                "에너지 다소비 기관을 중심으로 감축 여력을 재검토하고, "
                "설비 효율화 및 운영 개선 과제를 집중적으로 발굴할 필요가 있습니다."
            )
        else:
            paragraphs.append(
                "배출량이 기준배출량과 정확히 일치하는 수준으로 나타나고 있으며, "
                "향후 감축 목표를 상향 조정하거나 효율 개선 영역을 추가로 모색하는 것이 바람직합니다."
            )

    # 3. 최근 연도 추세
    trend_text = _analyze_trend_recent_years(recent_total_df, year)
    paragraphs.append(trend_text)

    # 4. 종합 코멘트
    paragraphs.append(
        "종합적으로, 공단 차원의 온실가스 관리 체계는 이행연도 배출 수준과 "
        "최근 연도별 추세를 함께 모니터링하면서, "
        "고배출 기관과 증가 추세 기관을 우선 관리 대상으로 선정하는 방향으로 운영하는 것이 바람직합니다."
    )

    return "\n\n".join(paragraphs)
