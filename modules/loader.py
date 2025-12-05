def build_data1_tables(df_raw_all: pd.DataFrame):
    """
    업로드 탭에서 사용하는 3개 표:
      1) 연도×기관 에너지 사용량(연단위)
      2) 연도×기관 연면적
      3) 연도별 3개년 평균 에너지 사용량 (직전 최대 3개년 평균)
    """
    df = df_raw_all.copy()

    years = sorted(df["연도"].unique())
    org_order = list(get_org_order())

    # 1) 연도×기관 에너지 사용량 (연단위)
    usage = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연단위",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=years)
        .reindex(columns=org_order)
    )
    usage["합계"] = usage.sum(axis=1)

    # 2) 연도×기관 연면적
    area = (
        df.pivot_table(
            index="연도",
            columns="기관명",
            values="연면적",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=years)
        .reindex(columns=org_order)
    )
    area["합계"] = area.sum(axis=1)

    # 3) 연도별 3개년 평균 에너지 사용량 (직전 최대 3개년 평균)
    avg3 = pd.DataFrame(index=years, columns=usage.columns, dtype=float)
    for y in years:
        prev_years = [py for py in years if py < y][-3:]
        if not prev_years:
            baseline = usage.loc[y]
        else:
            baseline = usage.loc[prev_years].mean()
        avg3.loc[y] = baseline

    def _reset_index_as_label(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out.insert(0, "구분", out.index.astype(str))
        return out.reset_index(drop=True)

    return (
        _reset_index_as_label(usage),
        _reset_index_as_label(area),
        _reset_index_as_label(avg3),
    )
