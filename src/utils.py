from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    import orjson as _orjson
except Exception:  # pragma: no cover - optional dependency
    _orjson = None


WORD_RE = re.compile(r"[a-z0-9][a-z0-9\+\#\.&/\-]*", re.IGNORECASE)

ENGINEERING_TITLE_HINTS = (
    "engineer",
    "developer",
    "scientist",
    "architect",
    "analyst",
    "data",
    "software",
    "backend",
    "frontend",
    "full stack",
    "ml",
    "machine learning",
    "ai",
    "devops",
    "qa",
)

NON_ENGINEERING_TITLE_HINTS = (
    "manager",
    "executive",
    "writer",
    "designer",
    "accountant",
    "hr",
    "recruiter",
    "sales",
    "support",
    "operations",
    "marketing",
    "consultant",
    "teacher",
    "assistant",
    "owner",
)

ROLE_CRITICAL_TERMS = (
    "python",
    "pytorch",
    "tensorflow",
    "embeddings",
    "retrieval",
    "ranking",
    "vector",
    "vector database",
    "milvus",
    "pinecone",
    "weaviate",
    "qdrant",
    "faiss",
    "elasticsearch",
    "opensearch",
    "llm",
    "fine-tuning",
    "finetuning",
    "lora",
    "qlora",
    "peft",
    "evaluation",
    "ndcg",
    "mrr",
    "map",
    "ab test",
    "a/b",
    "production",
    "deployed",
    "search",
    "recommendation",
    "nlp",
    "machine learning",
    "data pipeline",
    "spark",
    "airflow",
    "kafka",
    "docker",
    "kubernetes",
    "distributed systems",
)

PRODUCTION_TERMS = (
    "production",
    "deployed",
    "shipped",
    "users",
    "real-time",
    "latency",
    "scale",
    "monitoring",
    "regression",
    "refresh",
    "drift",
    "evaluation",
    "benchmark",
    "ab test",
    "offline",
    "online",
    "pipeline",
)

BUZZWORD_TERMS = (
    "langchain",
    "rag",
    "prompt engineering",
    "prompt",
    "openai api",
    "chatgpt",
    "demo",
    "tutorial",
    "bootcamp",
    "hackathon",
)

COMPANY_SIZE_SCORES = {
    "1-10": 0.55,
    "11-50": 0.68,
    "51-200": 0.78,
    "201-500": 0.88,
    "501-1000": 0.92,
    "1001-5000": 0.94,
    "5001-10000": 0.9,
    "10001+": 0.84,
}

EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.84,
    "tier_3": 0.72,
    "tier_4": 0.58,
    "unknown": 0.5,
}


def loads_json(data: bytes | str) -> Any:
    if _orjson is not None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _orjson.loads(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def dumps_json(data: Any) -> str:
    if _orjson is not None:
        return _orjson.dumps(data).decode("utf-8")
    return json.dumps(data, ensure_ascii=False)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s\+\#\.&/\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def count_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    haystack = normalize_text(text)
    return sum(1 for phrase in phrases if phrase in haystack)


def phrase_score(text: str, phrases: Iterable[str]) -> float:
    phrases = tuple(phrases)
    if not phrases:
        return 0.0
    haystack = normalize_text(text)
    hits = 0.0
    for phrase in phrases:
        if phrase in haystack:
            hits += 1.0
    return clamp(hits / float(len(phrases)))


def weighted_phrase_score(text: str, phrase_weights: dict[str, float]) -> float:
    if not phrase_weights:
        return 0.0
    haystack = normalize_text(text)
    total = 0.0
    hit = 0.0
    for phrase, weight in phrase_weights.items():
        total += weight
        if phrase in haystack:
            hit += weight
    return clamp(hit / total) if total else 0.0


def load_jsonl(path: str | Path):
    with open(path, "rb") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            yield loads_json(raw)


def read_docx_text(path: str | Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<w:p[^>]*>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def title_signal(title: str | None) -> dict[str, float]:
    text = normalize_text(title)
    engineering = 0.0
    non_engineering = 0.0
    ai_ml = 0.0

    for term in ENGINEERING_TITLE_HINTS:
        if term in text:
            engineering += 1.0
    for term in NON_ENGINEERING_TITLE_HINTS:
        if term in text:
            non_engineering += 1.0
    for term in ("ai engineer", "ml engineer", "machine learning", "data scientist", "research engineer"):
        if term in text:
            ai_ml += 1.0

    engineering = clamp(engineering / 4.0)
    non_engineering = clamp(non_engineering / 4.0)
    ai_ml = clamp(ai_ml / 2.0)
    return {
        "engineering": engineering,
        "non_engineering": non_engineering,
        "ai_ml": ai_ml,
        "score": clamp(0.65 * engineering + 0.35 * ai_ml - 0.45 * non_engineering),
    }


def years_score(years: float | int | None) -> float:
    if years is None:
        return 0.4
    years = float(years)
    center = 7.0
    width = 2.8
    return clamp(math.exp(-((years - center) ** 2) / (2.0 * width * width)))


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def skill_proficiency_score(proficiency: str | None) -> float:
    mapping = {
        "beginner": 0.3,
        "intermediate": 0.6,
        "advanced": 0.84,
        "expert": 1.0,
    }
    return mapping.get((proficiency or "").lower(), 0.5)


def expected_duration_weight(duration_months: int | float | None) -> float:
    if duration_months is None:
        return 0.0
    return clamp(float(duration_months) / 36.0)


def get_candidate_location_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    location = profile.get("location", "")
    country = profile.get("country", "")
    return f"{location} {country}".strip()
