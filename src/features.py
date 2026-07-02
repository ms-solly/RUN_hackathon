from __future__ import annotations

import math
from typing import Any

from .filters import location_relevance, score_penalties
from .utils import (
    COMPANY_SIZE_SCORES,
    EDUCATION_TIER_SCORES,
    PRODUCTION_TERMS,
    ROLE_CRITICAL_TERMS,
    clamp,
    count_phrase_hits,
    expected_duration_weight,
    normalize_text,
    phrase_score,
    safe_mean,
    skill_proficiency_score,
    title_signal,
    weighted_phrase_score,
    years_score,
)


def _candidate_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
        profile.get("location", ""),
        profile.get("country", ""),
    ]
    for item in history:
        parts.append(item.get("title", ""))
        parts.append(item.get("description", ""))
        parts.append(item.get("industry", ""))
        parts.append(item.get("company", ""))
    for item in education:
        parts.append(item.get("degree", ""))
        parts.append(item.get("field_of_study", ""))
        parts.append(item.get("institution", ""))
    for skill in skills:
        parts.append(skill.get("name", ""))
    for key, value in signals.get("skill_assessment_scores", {}).items():
        parts.append(f"{key} {value}")
    return " ".join(parts)


def _career_progression(candidate: dict[str, Any]) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 0.35

    durations = []
    seniority = 0.0
    prev_months = None
    promotion_bonus = 0.0
    for item in history:
        months = int(item.get("duration_months") or 0)
        durations.append(months)
        title = normalize_text(item.get("title", ""))
        if any(token in title for token in ("senior", "lead", "principal", "staff")):
            seniority += 1.0
        if any(token in title for token in ("engineer", "developer", "scientist", "analyst", "architect")):
            seniority += 0.4
        if prev_months is not None and months >= prev_months:
            promotion_bonus += 0.12
        prev_months = months
    stability = clamp((safe_mean(durations) - 6.0) / 18.0)
    diversity = clamp(len(history) / 6.0)
    seniority = clamp(seniority / 3.0)
    promotion_bonus = clamp(promotion_bonus)
    return clamp(0.34 * stability + 0.28 * diversity + 0.23 * seniority + 0.15 * promotion_bonus)


def _company_size_score(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    current_size = COMPANY_SIZE_SCORES.get(profile.get("current_company_size"), 0.55)
    history_sizes = []
    for item in history:
        history_sizes.append(COMPANY_SIZE_SCORES.get(item.get("company_size"), 0.55))
    if history_sizes:
        current_size = 0.55 * current_size + 0.45 * safe_mean(history_sizes)
    return clamp(current_size)


def _education_score(candidate: dict[str, Any]) -> float:
    education = candidate.get("education", [])
    if not education:
        return 0.45
    scores = []
    for item in education:
        tier = EDUCATION_TIER_SCORES.get(item.get("tier", "unknown"), 0.5)
        degree = normalize_text(item.get("degree", ""))
        field = normalize_text(item.get("field_of_study", ""))
        boost = 0.0
        if any(token in field for token in ("computer science", "engineering", "information technology", "data science", "electronics", "mathematics", "statistics")):
            boost += 0.18
        if any(token in degree for token in ("b.e", "b.tech", "m.e", "m.tech", "phd", "msc", "m.sc", "bsc", "mba")):
            boost += 0.08
        scores.append(clamp(tier + boost))
    return clamp(safe_mean(scores))


def _skill_trust(candidate: dict[str, Any], job_text: str) -> float:
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    if not skills:
        return 0.25

    values = []
    for skill in skills:
        name = skill.get("name", "")
        duration = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        proficiency = skill_proficiency_score(skill.get("proficiency"))
        duration_score = expected_duration_weight(duration)
        endorsement_score = clamp(math.log1p(endorsements) / 3.5)
        evidence = 0.25 + 0.35 * duration_score + 0.2 * endorsement_score + 0.2 * proficiency

        assessment = None
        for key, value in assessment_scores.items():
            if normalize_text(key) == normalize_text(name):
                assessment = float(value) / 100.0
                break
        if assessment is not None:
            evidence = 0.6 * evidence + 0.4 * assessment

        relevance = weighted_phrase_score(name, {term: 1.0 for term in ROLE_CRITICAL_TERMS})
        if relevance > 0:
            evidence *= 0.88 + 0.12 * relevance

        if duration <= 0 and skill.get("proficiency", "").lower() in {"advanced", "expert"}:
            evidence *= 0.58

        values.append(clamp(evidence))

    text = _candidate_text(candidate)
    relevant_skill_mentions = count_phrase_hits(text, ROLE_CRITICAL_TERMS)
    mention_bonus = clamp(relevant_skill_mentions / 14.0)
    return clamp(0.84 * safe_mean(values) + 0.16 * mention_bonus)


def _skill_assessment_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}
    if not assessment_scores:
        return 0.32

    scores = []
    for skill_name, value in assessment_scores.items():
        score = float(value) / 100.0
        relevance = phrase_score(skill_name, ROLE_CRITICAL_TERMS)
        if relevance > 0:
            score *= 0.85 + 0.15 * relevance
        scores.append(score)
    return clamp(safe_mean(scores))


def _career_relevance(candidate: dict[str, Any], job_text: str) -> float:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    candidate_text = _candidate_text(candidate)

    title_sig = title_signal(f"{profile.get('headline', '')} {profile.get('current_title', '')}")
    title_component = title_sig["score"]

    career_text = " ".join(f"{item.get('title', '')} {item.get('description', '')}" for item in history)
    career_matches = weighted_phrase_score(career_text, {term: 1.0 for term in ROLE_CRITICAL_TERMS})
    production_matches = weighted_phrase_score(career_text, {term: 1.0 for term in PRODUCTION_TERMS})
    jd_overlap = weighted_phrase_score(candidate_text, {term: 1.0 for term in ROLE_CRITICAL_TERMS})

    progression = _career_progression(candidate)
    company_size = _company_size_score(candidate)

    current_title = normalize_text(profile.get("current_title", ""))
    if any(term in current_title for term in ("manager", "writer", "designer", "accountant", "sales", "support", "operations")):
        title_component *= 0.86

    return clamp(0.28 * title_component + 0.30 * career_matches + 0.18 * production_matches + 0.16 * jd_overlap + 0.08 * progression * company_size)


def _experience_score(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    years = float(profile.get("years_of_experience") or 0.0)
    base = years_score(years)
    if 5.0 <= years <= 9.5:
        base = clamp(base + 0.12)
    elif years < 3.0:
        base *= 0.7
    elif years > 12.0:
        base *= 0.9
    return clamp(base)


def _behavior_feature(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    saved = float(signals.get("saved_by_recruiters_30d") or 0.0)
    interview_completion = float(signals.get("interview_completion_rate") or 0.0)
    notice_period = float(signals.get("notice_period_days") or 90.0)
    github = float(signals.get("github_activity_score") or 0.0)

    saved_score = clamp(math.log1p(saved) / 2.6)
    notice_score = 1.0 - clamp(abs(notice_period - 30.0) / 150.0)
    github_score = 0.0 if github < 0 else clamp(github / 100.0)

    return clamp(
        0.24 * open_to_work
        + 0.26 * response_rate
        + 0.14 * saved_score
        + 0.18 * interview_completion
        + 0.10 * notice_score
        + 0.08 * github_score
    )


def _github_score(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {})
    github = float(signals.get("github_activity_score") or -1.0)
    if github < 0:
        return 0.2
    return clamp(github / 100.0)


def extract_features(candidate: dict[str, Any], job_text: str) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    title_text = f"{profile.get('headline', '')} {profile.get('current_title', '')}"
    title_sig = title_signal(title_text)
    location = location_relevance(candidate)
    education = _education_score(candidate)
    experience = _experience_score(candidate)
    career_relevance = _career_relevance(candidate, job_text)
    skill_trust = _skill_trust(candidate, job_text)
    skill_assessment = _skill_assessment_score(candidate)
    behavior_feature = _behavior_feature(candidate)
    github = _github_score(candidate)
    company_size = _company_size_score(candidate)

    reasoning_skills = []
    assessments = signals.get("skill_assessment_scores", {}) or {}
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if weighted_phrase_score(name, {term: 1.0 for term in ROLE_CRITICAL_TERMS}) > 0:
            duration = int(skill.get("duration_months") or 0)
            if duration > 0:
                reasoning_skills.append(name)
    if not reasoning_skills:
        top_assessments = sorted(assessments.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:3]
        reasoning_skills = [k for k, _ in top_assessments]

    penalties = score_penalties(candidate, job_text)
    multiplier = 1.0
    multiplier *= 1.0 - 0.18 * penalties["buzz_penalty"]
    multiplier *= 1.0 - 0.20 * penalties["skill_penalty"]
    multiplier *= 1.0 - 0.16 * penalties["impossible_timeline_penalty"]
    multiplier *= 1.0 - 0.16 * penalties["mismatch_penalty"]
    multiplier *= 1.0 - 0.05 * penalties["company_penalty"]
    multiplier = clamp(multiplier, 0.4, 1.0)

    core = (
        0.35 * career_relevance
        + 0.20 * title_sig["score"]
        + 0.15 * skill_trust
        + 0.10 * skill_assessment
        + 0.08 * experience
        + 0.05 * behavior_feature
        + 0.03 * github
        + 0.02 * location
        + 0.02 * education
    )

    company_boost = 0.94 + 0.06 * company_size
    final_score = clamp(core * multiplier * company_boost)

    return {
        "title_score": title_sig["score"],
        "career_relevance": career_relevance,
        "skill_trust": skill_trust,
        "skill_assessment": skill_assessment,
        "experience": experience,
        "behavior_feature": behavior_feature,
        "github": github,
        "location": location,
        "education": education,
        "company_size": company_size,
        "penalties": penalties,
        "penalty_multiplier": multiplier,
        "final_score": final_score,
        "reasoning_skills": reasoning_skills,
        "assessment_names": sorted(assessments.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:4],
    }
