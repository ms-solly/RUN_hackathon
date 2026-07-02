from __future__ import annotations

from typing import Any

from .utils import clamp


def behavior_multiplier(candidate: dict[str, Any], features: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    saved = float(signals.get("saved_by_recruiters_30d") or 0.0)
    interview_completion = float(signals.get("interview_completion_rate") or 0.0)
    notice_period = float(signals.get("notice_period_days") or 90.0)
    github = float(signals.get("github_activity_score") or -1.0)
    github_score = 0.0 if github < 0 else clamp(github / 100.0)

    saved_bonus = clamp(saved / 6.0)
    notice_bonus = clamp(1.0 - abs(notice_period - 30.0) / 180.0)
    engagement = (
        0.25 * open_to_work
        + 0.25 * response_rate
        + 0.15 * saved_bonus
        + 0.20 * interview_completion
        + 0.08 * notice_bonus
        + 0.07 * github_score
    )

    boost = 0.92 + 0.16 * engagement
    if features.get("penalties", {}).get("buzz_penalty", 0.0) > 0.4:
        boost -= 0.04
    if features.get("penalties", {}).get("mismatch_penalty", 0.0) > 0.3:
        boost -= 0.03
    return clamp(boost, 0.85, 1.08)
