from __future__ import annotations

from typing import Any

from .behavioral import behavior_multiplier
from .features import extract_features
from .utils import clamp


def score_candidate(candidate: dict[str, Any], job_text: str) -> dict[str, Any]:
    features = extract_features(candidate, job_text)
    behavior = behavior_multiplier(candidate, features)
    final = clamp(features["final_score"] * behavior)
    features["behavior_multiplier"] = behavior
    features["final_score"] = final
    return features
