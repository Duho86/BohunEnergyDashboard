# modules/loader.py
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union, List, Dict, Any

import pandas as pd


class EnergyDataError(Exception):
    """에너지 사용량 데이터 처리 중 발생하는 공통 예외."""
    pass


# ===========================
# 경로/저장 관련 유틸
# ===========================

def ensure_energy_dir(base_dir: Union[str, Path] = "data/energy") -> Path:
    """
    data/energy 폴더가 없으면 생성하고, Path 객체를 반환한다.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def save_xlsx_file(
    file_obj: BinaryIO,
    original_filename: str,
    base_dir: Union[str, Path] = "data/energy",
) -> Path:
    """
    업로드된 .xlsx 파일을 data/energy/ 폴더에 저장한다.
    """
    energy_dir = ensure_energy_dir(base_dir)

    if not original_filename.lower().endswith(".xlsx"):
        raise EnergyDataError("지원하지 않는 파일 형식입니다. .xlsx 파일만 업로드해 주세요.")

    safe_name = os.path.basename(original_filename)
    dest_path = energy_dir / safe_name

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    try:
        data = file_obj.read()
    except Exception as e:
        raise EnergyDataError(f"업로드된 파일을 읽는 중 오류가 발생했습니다: {e}")

    if not data:
        raise EnergyDataError("업로드된 파일이 비어 있습니다.")

    with open(dest_path, "wb") as out:
        out.write(data)

    return dest_path


# ===========================
# 연도 인식 로직
# ===========================

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def detect_year_from_filename(filename: str) -> Optional[int]:
    match = YEAR_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(0))


def detect_year_from_dataframe(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None

    # '연도' 컬럼 우선
    for col in df.columns:
        if "연도" in str(col):
            year_series = df[col].dropna()
            if not year_series.empty:
                val = str(year_series.iloc[0])
                match = YEAR_PATTERN.search(val)
                if match:
                    return int(match.group(0))

    # 그 외 셀에서 연도 패턴 탐색
    sample = df.head(10)
    for col in sample.columns:
        for v in sample[col].astype(str):
            match = YEAR_PATTERN.search(v)
            if match:
                return int(match.group(0))

    return None


def detect_year(df: pd.DataFrame, filename: str) -> int:
    year = detect_year_from_filename(filename)
    if year is not None:
        return year

    year = detect_year_from_dataframe(df)
    if year is not None:
        return year

    raise EnergyDataError(
        f"연도를 인식할 수 없습니다. 파일명 또는 시트 내에 연도를 확인해 주세요. (filename={filename})"
    )


# ===========================
# 컬럼 식별/정규화 유틸
# ===========================

def _find_ghg_column(columns: List[str]) -> Optional[str]:
    """
    '온실가스'와 '환산'이 모두 포함된 컬럼명을 탐색.
    (예: '온실가스 환산량\\n(tCO2eq)')
    """
    for col in columns:
        normalized = str(col).replace("\n", "")
        if "온실가스" in normalized and "환산" in normalized:
            return col
    return None


def _find_facility_column(columns: List[str]) -> Optional[str]:
    """
    기관/시설 이름 컬럼을 탐색.
    """
    for col in columns:
        c = str(col)
        if "기관명" in c:
            return col
    for col in columns:
        c = str(col)
        if "시설명" in c:
            return col
    for col in columns:
        c = str(col)
        if "시설내역" in c:
            return col
    return None


def _find_month_columns(columns: List[str]) -> List[str]:
    """
    '에너지'와 '사용량'이 모두 포함된 월별 사용량 컬럼들을 탐색.
    (예: '에너지사용량', '에너지사용량.1', ...)
    """
    month_cols: List[str] = []
    for col in columns:
        c = str(col)
        if "에너지" in c and "사용량" in c:
            month_cols.append(col)
    return month_cols


# ===========================
# 숫자 컬럼 전처리 유틸
# ===========================

_PLACEHOLDER_VALUES = {"", " ", "-", "--", "—", "N/A", "NA", "NaN", "nan"}


def clean_numeric_series(
    s: pd.Series,
    treat_placeholders_as_zero: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    숫자로 계산해야 하는 Series를 안전하게 float로 변환한다.

    1) "", 공백, "-", N/A 등 placeholder 값 처리
    2) pd.to_numeric(errors="coerce") 로 강제 변환
    3) 숫자로 변환 실패한 위치의 bool mask 반환

    반환:
        (cleaned_series, invalid_mask)
    """
    # 우선 문자열로 통일
    s_obj = s.astype("object")

    # placeholder 처리
    mask_placeholder = s_obj.isin(_PLACEHOLDER_VALUES)
    if treat_placeholders_as_zero:
        s_obj = s_obj.mask(mask_placeholder, 0)
    else:
        s_obj = s_obj.mask(mask_placeholder, pd.NA)

    # 숫자 변환
    s_num = pd.to_numeric(s_obj, errors="coerce")

    # 변환 실패한 위치 (placeholder 제외, 진짜 이상값)
    invalid_mask = s_obj.notna() & s_num.isna()

    return s_num.astype("float64"), invalid_mask


# ===========================
# 엑셀 구조 사전 진단 함수
# ===========================

def validate_excel_structure(
    df_raw: pd.DataFrame,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    엑셀 시트 구조를 사전 점검한다.

    반환 값:
        {
          "ok": bool,
          "issues": [str, ...],    # 반드시 수정해야 하는 문제(업로드 차단)
          "warnings": [str, ...],  # 참고용 경고(업로드 허용)
          "detected_facility_col": Optional[str],
          "detected_ghg_col": Optional[str],
          "detected_month_cols": List[str],
          "filename": Optional[str],
        }
    """
    issues: List[str] = []
    warnings: List[str] = []

    if df_raw is None or df_raw.empty:
        issues.append("시트에 데이터가 없거나, 모든 행이 비어 있습니다.")
        return {
            "ok": False,
            "issues": issues,
            "warnings": warnings,
            "detected_facility_col": None,
            "detected_ghg_col": None,
            "detected_month_cols": [],
            "filename": filename,
        }

    columns = list(df_raw.columns)
    col_names = [str(c) for c in columns]

    facility_col = _find_facility_column(columns)
    ghg_col = _find_ghg_column(columns)
    month_cols = _find_month_columns(columns)

    # 1) 필수 컬럼 존재 여부 (누락 시 업로드 차단)
    if facility_col is None:
        issues.append(
            "기관/시설명을 나타내는 컬럼을 찾을 수 없습니다. "
            "예상 컬럼명 예시: '기관명', '시설명', '시설내역'. "
            f"현재 컬럼: {col_names}"
        )

    if ghg_col is None:
        issues.append(
            "온실가스 환산량 컬럼을 찾을 수 없습니다. "
            "예상: '온실가스 환산량\\n(tCO2eq)' 등 '온실가스'와 '환산'이 모두 포함된 컬럼. "
            f"현재 컬럼: {col_names}"
        )

    if not month_cols:
        issues.append(
            "월별 에너지 사용량 컬럼을 찾을 수 없습니다. "
            "예상: 이름에 '에너지'와 '사용량'이 모두 포함된 1~12개 컬럼 "
            "(예: '에너지사용량', '에너지사용량.1' 등). "
            f"현재 컬럼: {col_names}"
        )
    elif len(month_cols) < 12:
        warnings.append(
            f"월별 에너지 사용량으로 추정되는 컬럼이 {len(month_cols)}개만 발견되었습니다. "
            f"(발견된 컬럼: {month_cols}) 실제 12개월이 모두 포함되었는지 확인해 주세요."
        )

    # 2) 숫자형 컬럼의 이상값 점검 (업로드는 허용, 경고만 노출)
    # 온실가스 환산량
    if ghg_col is not None and ghg_col in df_raw.columns:
        s = df_raw[ghg_col]
        _, invalid_mask = clean_numeric_series(s, treat_placeholders_as_zero=True)
        invalid_cnt = int(invalid_mask.sum())
        if invalid_cnt > 0:
            sample_vals = s[invalid_mask].astype(str).head(5).tolist()
            warnings.append(
                f"온실가스 환산량 컬럼('{ghg_col}')에 숫자로 변환할 수 없는 값이 "
                f"{invalid_cnt}개 있습니다. 해당 값은 계산 시 NaN으로 처리됩니다. "
                f"예시 값: {sample_vals}"
            )

    # 월별 에너지 사용량 컬럼들
    invalid_month_msgs: List[str] = []
    for mc in month_cols:
        if mc not in df_raw.columns:
            continue
        s = df_raw[mc]
        _, invalid_mask = clean_numeric_series(s, treat_placeholders_as_zero=True)
        invalid_cnt = int(invalid_mask.sum())
        if invalid_cnt > 0:
            invalid_month_msgs.append(f"'{mc}' (이상값 {invalid_cnt}개)")

    if invalid_month_msgs:
        warnings.append(
            "다음 월별 에너지 사용량 컬럼에 숫자로 변환할 수 없는 값이 있습니다. "
            "해당 값은 계산 시 NaN으로 처리됩니다: "
            + ", ".join(invalid_month_msgs)
        )

    ok = len(issues) == 0

    return {
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "detected_facility_col": facility_col,
        "detected_ghg_col": ghg_col,
        "detected_month_cols": month_cols,
        "filename": filename,
    }


def validate_excel_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    개별 엑셀 파일(path)에 대해 시트 구조를 진단한다.
    (디버깅/진단 탭에서 사용)
    """
    path = Path(path)
    if not path.exists():
        return {
            "ok": False,
            "issues": [f"파일을 찾을 수 없습니다: {path}"],
            "warnings": [],
            "detected_facility_col": None,
            "detected_ghg_col": None,
            "detected_month_cols": [],
            "filename": path.name,
        }

    try:
        df_raw = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        return {
            "ok": False,
            "issues": [f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}"],
            "warnings": [],
            "detected_facility_col": None,
            "detected_ghg_col": None,
            "detected_month_cols": [],
            "filename": path.name,
        }

    return validate_excel_structure(df_raw, filename=path.name)


# ===========================
# 정규화(melt) 로직
# ===========================

def normalize_energy_dataframe(
    df_raw: pd.DataFrame,
    year: int,
    source_filename: str,
) -> pd.DataFrame:
    """
    원본 엑셀(df_raw)을 표준 스키마(연도/기관명/월/에너지사용량/온실가스 환산량)로 변환.
    """
    if df_raw is None or df_raw.empty:
        raise EnergyDataError("엑셀 시트에 데이터가 없거나, 모두 비어 있습니다.")

    columns = list(df_raw.columns)

    facility_col = _find_facility_column(columns)
    if facility_col is None:
        raise EnergyDataError(
            "기관/시설을 나타내는 컬럼을 찾을 수 없습니다. "
            "(예: '기관명', '시설명', '시설내역')"
        )

    ghg_col = _find_ghg_column(columns)
    if ghg_col is None:
        raise EnergyDataError(
            "['온실가스 환산량'] 컬럼을 찾을 수 없습니다. "
            "(예: '온실가스 환산량\\n(tCO2eq)')"
        )

    month_cols = _find_month_columns(columns)
    if not month_cols:
        raise EnergyDataError(
            "월별 에너지 사용량 컬럼을 찾을 수 없습니다. "
            "(예: '에너지사용량', '에너지사용량.1' 등)"
        )

    id_vars = [c for c in [facility_col, ghg_col] if c in df_raw.columns]

    # value_name 이 기존 컬럼명과 겹치지 않도록 임시 이름 사용
    value_tmp_col = "__energy_value__"
    while value_tmp_col in df_raw.columns:
        value_tmp_col += "_x"

    try:
        df_melted = df_raw.melt(
            id_vars=id_vars,
            value_vars=month_cols,
            var_name="month_col",
            value_name=value_tmp_col,  # 임시 이름
        )
    except Exception as e:
        raise EnergyDataError(f"월별 데이터 구조를 변환하는 중(melt) 오류가 발생했습니다: {e}")

    # month_col → 월 번호(1~12) 매핑
    month_map = {col: idx for idx, col in enumerate(month_cols, start=1)}
    df_melted["월"] = df_melted["month_col"].map(month_map)

    # 월 정보가 없는 행 제거
    df_melted = df_melted.dropna(subset=["월"])

    # 중간 컬럼 제거
    df_melted = df_melted.drop(columns=["month_col"])

    # 컬럼명 표준화
    df_melted = df_melted.rename(
        columns={
            facility_col: "기관명",
            ghg_col: "온실가스 환산량",
            value_tmp_col: "에너지사용량",
        }
    )

    # 🔢 숫자 컬럼 전처리: 에너지사용량, 온실가스 환산량
    energy_clean, energy_invalid = clean_numeric_series(
        df_melted["에너지사용량"], treat_placeholders_as_zero=True
    )
    ghg_clean, ghg_invalid = clean_numeric_series(
        df_melted["온실가스 환산량"], treat_placeholders_as_zero=True
    )

    df_melted["에너지사용량"] = energy_clean
    df_melted["온실가스 환산량"] = ghg_clean

    # 전처리 결과(이상값 개수)를 attrs 로 남겨두면 필요시 디버깅에 활용 가능
    df_melted.attrs["invalid_energy_count"] = int(energy_invalid.sum())
    df_melted.attrs["invalid_ghg_count"] = int(ghg_invalid.sum())

    # 연도 및 파일명 컬럼 추가
    df_melted["연도"] = int(year)
    df_melted["source_file"] = source_filename

    # 완전 비어 있는 행 제거
    df_melted = df_melted.dropna(subset=["에너지사용량", "기관명"], how="all")

    if df_melted.empty:
        raise EnergyDataError("정규화 후 남은 유효 데이터가 없습니다. 엑셀 내용을 다시 확인해 주세요.")

    df_std = df_melted[["연도", "기관명", "월", "에너지사용량", "온실가스 환산량", "source_file"]]

    return df_std


# ===========================
# 공개용 상위 함수
# ===========================

def load_energy_xlsx(
    path: Union[str, Path],
) -> Tuple[pd.DataFrame, int]:
    """
    1) 엑셀 읽기
    2) 구조 사전 진단 (validate_excel_structure)
    3) 연도 인식
    4) 표준 스키마 정규화
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    try:
        df_raw = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        raise EnergyDataError(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")

    # 1단계: 엑셀 구조 사전 진단
    validation = validate_excel_structure(df_raw, filename=path.name)

    # 필수 구조 문제(issue)가 있으면 업로드 중단
    if not validation["ok"]:
        issue_lines = "\n".join(f"- {msg}" for msg in validation["issues"])
        raise EnergyDataError(
            "엑셀 구조 점검에서 다음 문제가 발견되었습니다. "
            "엑셀 양식을 수정한 후 다시 업로드해 주세요.\n" + issue_lines
        )

    # 경고(warning)는 업로드 허용 + 사용자에게만 알림 (app.py의 디버그 탭에서 확인 가능)

    # 2단계: 연도 인식
    year = detect_year(df_raw, path.name)

    # 3단계: 표준 스키마 변환
    df_std = normalize_energy_dataframe(df_raw, year=year, source_filename=path.name)

    # 필수 컬럼 최종 검증
    required_cols = {"기관명", "월", "온실가스 환산량", "에너지사용량"}
    missing = required_cols - set(df_std.columns)
    if missing:
        raise EnergyDataError(f"표준 스키마에서 필수 컬럼이 누락되었습니다: {missing}")

    return df_std, year


def process_uploaded_energy_file(
    file_obj: BinaryIO,
    original_filename: str,
    base_dir: Union[str, Path] = "data/energy",
) -> Tuple[pd.DataFrame, int, Path]:
    """
    Streamlit 업로드 파일을 저장 → 구조 점검 → 표준 스키마 정규화까지 수행.
    """
    saved_path = save_xlsx_file(file_obj, original_filename, base_dir=base_dir)
    df_std, year = load_energy_xlsx(saved_path)
    return df_std, year, saved_path
