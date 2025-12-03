# modules/baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


DEFAULT_BASELINE_PATH = Path("data") / "baseline.json"


def load_baseline_records(path: Path = DEFAULT_BASELINE_PATH) -> Dict[int, Dict[str, Any]]:
    """
    baseline.json 에서 연도별 기준배출량 정보를 읽어온다.

    파일 형식 예시:
    {
      "2021": { "baseline": 1234.0, "is_target": false },
      "2022": { "baseline": 1300.0, "is_target": true }
    }

    과거 버전처럼 값만 저장되어 있을 수도 있으므로 모두 처리한다.
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    records: Dict[int, Dict[str, Any]] = {}

    for year_str, rec in raw.items():
        try:
            year = int(year_str)
        except Exception:
            continue

        baseline_val = None
        is_target = False

        if isinstance(rec, dict):
            baseline_val = rec.get("baseline")
            is_target = bool(rec.get("is_target", False))
        else:
            # 과거: "2021": 1234.0 이런 형태였던 경우
            baseline_val = rec

        try:
            baseline_val = float(baseline_val) if baseline_val is not None else None
        except Exception:
            baseline_val = None

        records[year] = {
            "baseline": baseline_val,
            "is_target": is_target,
        }

    return records


def save_baseline_records(records: Dict[int, Dict[str, Any]], path: Path = DEFAULT_BASELINE_PATH) -> None:
    """
    연도별 기준배출량 정보를 baseline.json 에 저장한다.
    """
    serializable: Dict[str, Any] = {}
    for year, rec in records.items():
        year_str = str(year)
        baseline_val = rec.get("baseline")
        is_target = bool(rec.get("is_target", False))
        serializable[year_str] = {
            "baseline": baseline_val,
            "is_target": is_target,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def get_baseline_map(records: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
    """
    analyzer/전망분석에서 사용할 수 있도록
    {연도: 기준배출량(float)} 딕셔너리로 변환.
    """
    baseline_map: Dict[int, float] = {}
    for year, rec in records.items():
        baseline = rec.get("baseline")
        if baseline is None:
            continue
        try:
            baseline_map[year] = float(baseline)
        except Exception:
            continue
    return baseline_map
