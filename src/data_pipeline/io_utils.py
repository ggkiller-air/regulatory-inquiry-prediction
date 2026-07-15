from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_csv(path: Path, rows: Iterable[BaseModel | dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.model_dump() if isinstance(row, BaseModel) else row
            writer.writerow({field: data.get(field, "") for field in fieldnames})


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            data = row.model_dump() if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

