from __future__ import annotations

from typing import Any


def _format_list(items: list[str], limit: int = 3) -> str:
    clean = [item for item in items if item]
    if not clean:
        return ""
    clean = clean[:limit]
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{clean[0]}, {clean[1]} and {clean[2]}"


def build_reasoning(candidate: dict[str, Any], features: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    title = profile.get("current_title") or profile.get("headline") or "candidate"
    years = profile.get("years_of_experience")
    years_text = f"{years:.1f} years" if isinstance(years, (int, float)) else "relevant experience"

    skill_names = features.get("reasoning_skills") or []
    if not skill_names:
        skill_names = [name for name, _ in features.get("assessment_names", [])][:3]
    skill_text = _format_list(skill_names)

    assessment_names = []
    for name, value in features.get("assessment_names", [])[:3]:
        if float(value) >= 0:
            assessment_names.append(f"{name} assessment {float(value):.0f}")
    assessment_text = _format_list(assessment_names)

    behavior_bits = []
    if signals.get("open_to_work_flag"):
        behavior_bits.append("open to work")
    response_rate = signals.get("recruiter_response_rate")
    if isinstance(response_rate, (int, float)):
        behavior_bits.append(f"recruiter response rate {response_rate:.2f}")
    github = signals.get("github_activity_score")
    if isinstance(github, (int, float)) and github >= 0:
        behavior_bits.append(f"GitHub activity {github:.1f}")
    notice = signals.get("notice_period_days")
    if isinstance(notice, int):
        behavior_bits.append(f"notice period {notice} days")
    behavior_text = _format_list(behavior_bits, limit=2)

    location = profile.get("location") or profile.get("country")
    location_text = f"based in {location}" if location else ""

    parts = []
    parts.append(f"{title} with {years_text}")
    if skill_text:
        parts.append(f"relevant skills in {skill_text}")
    if assessment_text:
        parts.append(f"strong {assessment_text}")
    if behavior_text:
        parts.append(behavior_text)
    if location_text:
        parts.append(location_text)

    reasoning = "; ".join(parts)
    return reasoning[:280]
