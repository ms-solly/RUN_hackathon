from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .utils import load_jsonl, read_docx_text


def load_candidates(path: str | Path):
    return load_jsonl(path)


def load_job_description(path: str | Path) -> str:
    return read_docx_text(path)


def write_submission(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": f"{row['score']:.6f}",
                    "reasoning": row["reasoning"],
                }
            )
