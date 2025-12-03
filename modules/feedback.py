# modules/feedback.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional
import pandas as pd


# ============================================================
# 내부: 최근 5개년 트렌드 분석
# ============================================================

def _analyze_trend_recent_years(recent_df: Optional[pd.DataFrame], current_year: int) -> str:
    """
    최근 연도 에너지 사용량(온실가스 배출량) 추세를 분석하여
    간단한 한국어 문장으로 요약.
    """
    if recent_df is None or recent_df.empty:
        return "최근 연도별 배출량 추세를 분석할 수 있는 데이터가 부족합니다."

    df = recent_df.copy()

    if "연도" not in df.columns or "연간 온실가스 배출량" not in df.columns:
        return "최근 연도별 배출량 추세를 분석할 수 있는 데이터가 부족합니다."

    df = df.sort_values("연도").reset_index(drop=True)

    cur = df[df["연도"] == current_year]

    if cur.empty:
        # 전체 범위로 단순 증가/감소 판단
        start_year = int(df["연도"].iloc[0])
        end_year = int(df["연도"].iloc[-1])
        first_val = df["연간 온실가스 배출량"].iloc[0]
        last_val = df["연간 온실가스 배출량"].iloc[-1]

        diff = last_val - first_val
        pct = diff / first_val * 100 if first_val else 0

        if diff > 0:
            return f"{start_year}년부터 {end_year}년까지 배출량은 약 {pct:,.1f}% 증가하는 추세입니다."
        elif diff < 0:
            return f"{start_year}년부터 {end_year}년까지 배출량은 약 {abs(pct):,.1f}% 감소하는 추세입니다."
        else:
            return f"{start_year}년부터 {end_year}년까지 배출량은 큰 변화 없이 유사한 수준을 유지하고 있습니다."

    # 현재 연도 존재
    cur_val = float(cur["연간 온실가스 배출량"].iloc[0])
    prev_year = current_year - 1
    prev = df[df["연도"] == prev_year]

    sentences = []

    if not prev.empty:
        prev_val = float(prev["연간 온실가스 배출량"].iloc[0])
        diff = cur_val - prev_val
        pct = diff / prev_val * 100 if prev_val else 0

        if diff > 0:
            sentences.append(
                f"{current_year}년 연간 배출량은 전년 대비 약 {pct:,.1f}%({diff:,.0f} tCO2eq) 증가했습니다."
            )
        elif diff < 0:
            sentences.append(
                f"{current_year}년 연간 배출량은 전년 대비 약 {abs(pct):,.1f}%({abs(diff):,.0f} tCO2eq) 감소했습니다."
            )
        else:
            sentences.append(f"{current_year}년 배출량은 전년과 거의 동일한 수준입니다.")
    else:
        sentences.append(
            f"{current_year}년 배출량은 {cur_val:,.0f} tCO2eq로 집계되었으며 전년 데이터는 존재하지 않습니다."
        )

    # 최대/최소 위치 기반 추가 분석
    max_row = df.loc[df["연간 온실가스 배출량"].idxmax()]
    min_row = df.loc[df["연간 온실가스 배출량"].idxmin()]

    max_y, min_y = int(max_row["연도"]), int(min_row["연도"])
    max_v, min_v = float(max_row["연간 온실가스 배출량"]), float(min_row["연간 온실가스 배출량"])

    if current_year == max_y:
        sentences.append(
            f"또한, 최근 분석 기간 중 {current_year}년이 가장 높은 배출량을 기록하고 있습니다."
        )
    elif current_year == min_y:
        sentences.append(
            f"또한, 최근 분석 기간 중 {current_year}년이 가장 낮은 배출량을 기록하고 있어 감축 정책의 효과가 반영된 것으로 판단됩니다."
        )
    else:
        mid = (max_v + min_v) / 2
        if cur_val > mid:
            sentences.append(
                f"{current_year}년 배출 수준은 최근 연도 중 상위권에 해당합니다."
            )
        else:
            sentences.append(
                f"{current_year}년 배출 수준은 최근 연도 중 중간 이하 수준입니다."
            )

    return " ".join(sentences)


# ============================================================
# 공단 전체 피드백 문단 생성 (유지)
# ============================================================

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
    공단 전체 연간 배출량 + 기준배출량 대비 해석 문단
    기존 보고서 스타일 문장 유지 (요청 반영)
    """
    paragraphs = []

    # -------------------------
    # 1. 기본 현황
    # -------------------------

    if actual_emission is None or pd.isna(actual_emission):
        paragraphs.append(
            f"{year}년 온실가스 배출량은 아직 집계되지 않아 분석이 어렵습니다."
        )
        paragraphs.append(_analyze_trend_recent_years(recent_total_df, year))
        return "\n\n".join(paragraphs)

    if current_month:
        paragraphs.append(
            f"{year}년 {current_month}월 기준 공단 전체 온실가스 배출량은 "
            f"{actual_emission:,.0f} tCO2eq입니다."
        )
    else:
        paragraphs.append(
            f"{year}년 공단 전체 온실가스 배출량은 "
            f"{actual_emission:,.0f} tCO2eq로 집계되었습니다."
        )

    # -------------------------
    # 2. 기준배출량 대비 분석
    # -------------------------

    if baseline_emission is None or pd.isna(baseline_emission) or baseline_emission == 0:
        paragraphs.append(
            "해당 연도 기준배출량이 없어 목표 대비 이행수준 분석은 불가합니다."
        )
    else:
        if ratio_to_baseline is not None:
            paragraphs.append(
                f"기준배출량은 {baseline_emission:,.0f} tCO2eq이며 "
                f"현재 배출량은 기준의 약 {ratio_to_baseline*100:,.1f}% 수준입니다."
            )
        else:
            paragraphs.append(
                f"기준배출량은 {baseline_emission:,.0f} tCO2eq로 설정되어 있습니다."
            )

        if reduction_rate_pct is not None:
            if reduction_rate_pct > 0:
                paragraphs.append(
                    f"기준 대비 약 {reduction_rate_pct:,.1f}% 감축된 상태입니다."
                )
            elif reduction_rate_pct < 0:
                paragraphs.append(
                    f"기준 대비 약 {abs(reduction_rate_pct):,.1f}% 초과 배출된 상태로 관리 강화가 필요합니다."
                )
            else:
                paragraphs.append(
                    "기준배출량과 동일한 수준으로 배출하고 있습니다."
                )

        diff = baseline_emission - actual_emission
        if diff > 0:
            paragraphs.append(
                f"현재 배출량은 기준보다 약 {diff:,.0f} tCO2eq 낮아 목표를 초과 달성하고 있습니다."
            )
        elif diff < 0:
            paragraphs.append(
                f"기준 대비 약 {abs(diff):,.0f} tCO2eq 초과 배출 상태입니다."
            )
        else:
            paragraphs.append("배출량이 기준과 정확히 동일합니다.")

    # -------------------------
    # 3. 최근 연도 추세 분석
    # -------------------------

    paragraphs.append(_analyze_trend_recent_years(recent_total_df, year))

    # -------------------------
    # 4. 종합 코멘트
    # -------------------------

    paragraphs.append(
        "종합적으로, 공단 차원의 온실가스 관리는 최근 연도별 추세와 "
        "기관별 에너지 사용 패턴을 함께 고려하여 우선 관리 대상을 설정하는 방식이 바람직합니다."
    )

    return "\n\n".join(paragraphs)
