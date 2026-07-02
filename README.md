# Intelligent Candidate Discovery Ranker

## Problem

Rank the top 100 candidates from `candidates.jsonl` for the supplied senior AI engineer job description. The dataset is intentionally adversarial: many profiles are non-engineering, some are keyword-stuffed, and a small fraction contain genuine AI/ML experience.

## Architecture

The solution is a local, rule-driven ranking pipeline with four stages:

1. Candidate filtering and penalty modeling for keyword stuffing, impossible timelines, title mismatch, and shallow AI claims.
2. Feature extraction across title relevance, career relevance, skill trust, skill assessment, experience, education, location, company size, and behavior.
3. Weighted scoring with a bounded behavioral multiplier.
4. Ranking, deterministic score assignment, and reasoning generation for the final submission.

## Feature Engineering

The ranker favors real production ML profiles over buzzword-heavy resumes by combining:

- title and career-history relevance to the target role,
- evidence-backed skill trust using duration, endorsements, and assessments,
- production signals such as deployment, monitoring, evaluation, and scale,
- soft experience shaping around the 5-9 year target band,
- recruiter engagement and availability signals,
- light education and company-size modifiers.

Keyword stuffing is penalized when AI terms appear without corresponding duration, assessment, or production evidence.

## Scoring Methodology

The final score is a normalized weighted sum of the major components:

- Career / title: 35%
- Career descriptions: 20%
- Skill trust: 15%
- Skill assessment: 10%
- Experience: 8%
- Behavior: 5%
- GitHub: 3%
- Location: 2%
- Education: 2%

The score is further adjusted by a bounded behavior multiplier and anti-stuffing penalties. The final submission uses monotonically decreasing output scores to satisfy validation.

[!NOTE] To test this prototype you need to 1. clone the repo => upload datasets from [here](https://drive.google.com/file/d/1MfD47XvVdRKBGRAyzGOxDCEf2ve96Jjo/view) => then test run 

## Running Instructions

```bash
python rank.py --candidates ./candidates.jsonl --job-description ./job_description.docx --out ./submission.csv
python validate_submission.py submission.csv
```


## Folder Structure

```text
src/
  io.py
  filters.py
  features.py
  behavioral.py
  scoring.py
  reasoning.py
  rank.py
  utils.py
rank.py
requirements.txt
submission_metadata.yaml
README.md
```

## Tradeoffs

- This is a deterministic POC, so it is fast and reproducible, but it cannot learn new patterns from labeled data.
- Hand-crafted relevance vocabularies are strong against the adversarial setting, but they will not generalize as well as a trained reranker on a broad corpus.
- The system intentionally avoids external APIs and embeddings to keep runtime local and competition-safe.

## Future Improvements

- Add offline-trained reranking with labeled pairs from recruiter feedback.
- Replace string matching with local embeddings and learned similarity features.
- Calibrate weights with validation data if labels become available.
- Add more robust extraction for career progression and title seniority.
