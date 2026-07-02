from __future__ import annotations

from typing import Any

from .utils import (
    BUZZWORD_TERMS,
    clamp,
    count_phrase_hits,
    get_candidate_location_text,
    normalize_text,
    title_signal,
)


def _career_months(candidate: dict[str, Any]) -> int:
    history = candidate.get("career_history") or []
    total = 0
    for item in history:
        try:
            total += int(item.get("duration_months") or 0)
        except Exception:
            continue
    return total


def score_penalties(candidate: dict[str, Any], job_text: str) -> dict[str, float]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    title_text = f"{profile.get('headline', '')} {profile.get('current_title', '')}"
    title_sig = title_signal(title_text)
    full_text_parts = [title_text, profile.get("summary", ""), job_text]
    for item in history:
        full_text_parts.append(item.get("title", ""))
        full_text_parts.append(item.get("description", ""))
    for skill in skills:
        full_text_parts.append(skill.get("name", ""))
    full_text = " ".join(full_text_parts)

    buzz_hits = count_phrase_hits(full_text, BUZZWORD_TERMS)
    buzz_penalty = clamp(max(0.0, buzz_hits - 1) / 6.0)

    expert_zero_duration = 0
    short_expert_claims = 0
    for skill in skills:
        duration = int(skill.get("duration_months") or 0)
        proficiency = normalize_text(skill.get("proficiency", ""))
        if duration <= 0 and proficiency in {"advanced", "expert"}:
            expert_zero_duration += 1
        if duration < 3 and proficiency == "expert":
            short_expert_claims += 1

    skill_penalty = clamp((expert_zero_duration * 0.14) + (short_expert_claims * 0.12))

    career_months = _career_months(candidate)
    years = float(profile.get("years_of_experience") or 0.0)
    claimed_months = int(round(years * 12.0))
    timeline_gap = max(0, claimed_months - career_months)
    impossible_timeline_penalty = clamp(timeline_gap / 84.0)

    engineering_title = title_sig["engineering"] > 0.45 or title_sig["ai_ml"] > 0.35
    non_engineering_title = title_sig["non_engineering"] > 0.4
    ai_signal = count_phrase_hits(full_text, ("embeddings", "retrieval", "ranking", "vector", "llm", "pytorch", "python", "fine-tuning", "evaluation", "nlp"))
    mismatch_penalty = 0.0
    if non_engineering_title and ai_signal <= 2:
        mismatch_penalty += 0.24
    if non_engineering_title and buzz_hits >= 3 and ai_signal <= 4:
        mismatch_penalty += 0.18
    if engineering_title and ai_signal == 0:
        mismatch_penalty += 0.08

    company_size = profile.get("current_company_size")
    company_penalty = 0.0 if company_size else 0.04
    no_recent_activity = 0.06 if signals.get("open_to_work_flag") is False else 0.0

    return {
        "buzz_penalty": buzz_penalty,
        "skill_penalty": skill_penalty,
        "impossible_timeline_penalty": impossible_timeline_penalty,
        "mismatch_penalty": clamp(mismatch_penalty),
        "company_penalty": clamp(company_penalty),
        "no_recent_activity": clamp(no_recent_activity),
    }


def penalty_multiplier(penalties: dict[str, float]) -> float:
    multiplier = 1.0
    multiplier *= 1.0 - 0.38 * penalties.get("buzz_penalty", 0.0)
    multiplier *= 1.0 - 0.30 * penalties.get("skill_penalty", 0.0)
    multiplier *= 1.0 - 0.26 * penalties.get("impossible_timeline_penalty", 0.0)
    multiplier *= 1.0 - 0.24 * penalties.get("mismatch_penalty", 0.0)
    multiplier *= 1.0 - 0.10 * penalties.get("company_penalty", 0.0)
    multiplier *= 1.0 - 0.05 * penalties.get("no_recent_activity", 0.0)
    return clamp(multiplier, 0.35, 1.0)


def location_relevance(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location_text = get_candidate_location_text(candidate)
    text = normalize_text(location_text)

    score = 0.45
    if profile.get("country", "").strip().lower() == "india":
        score += 0.2
    if any(city in text for city in ("pune", "noida", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai")):
        score += 0.18
    if signals.get("willing_to_relocate"):
        score += 0.1
    mode = (signals.get("preferred_work_mode") or "").lower()
    if mode in {"hybrid", "flexible", "onsite"}:
        score += 0.05
    return clamp(score)
