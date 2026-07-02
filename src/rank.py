from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import load_candidates, load_job_description, write_submission
from .reasoning import build_reasoning
from .scoring import score_candidate
from .utils import clamp


DEFAULT_CANDIDATES = Path("candidates.jsonl")
DEFAULT_JOB_DESCRIPTION = Path("job_description.docx")
DEFAULT_OUT = Path("submission.csv")


def rank_candidates(candidates_path: str | Path, job_description_path: str | Path, out_path: str | Path, top_n: int = 100) -> list[dict[str, Any]]:
    job_text = load_job_description(job_description_path)
    scored: list[tuple[float, str]] = []

    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if not cid:
            continue
        features = score_candidate(candidate, job_text)
        scored.append((float(features["final_score"]), cid))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:top_n]
    selected_ids = {cid for _, cid in selected}

    selected_candidates: dict[str, dict[str, Any]] = {}
    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if cid in selected_ids:
            selected_candidates[cid] = candidate
        if len(selected_candidates) == len(selected_ids):
            break

    rows: list[dict[str, Any]] = []
    for rank, (raw_score, cid) in enumerate(selected, start=1):
        candidate = selected_candidates[cid]
        features = score_candidate(candidate, job_text)
        display_score = round(1.0 - 0.00091 * (rank - 1) + 0.000001 * clamp(features["final_score"]), 6)
        rows.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "score": display_score,
                "reasoning": build_reasoning(candidate, features),
            }
        )

    write_submission(out_path, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank top candidates for the Redrob challenge.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="Path to candidates.jsonl")
    parser.add_argument("--job-description", default=str(DEFAULT_JOB_DESCRIPTION), help="Path to job_description.docx")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to submission.csv")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to include")
    args = parser.parse_args(argv)

    rank_candidates(args.candidates, args.job_description, args.out, args.top_n)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
