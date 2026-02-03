# Temporal Dynamics Benchmark Results

*Generated: 2026-02-03T12:28:18.310932*

## Summary

| Metric | Value |
|--------|-------|
| **System** | engram |
| **Total Cases** | 200 |
| **Overall Accuracy** | 80.0% |

## Results by Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| recency_override | 30 | 50 | 60.0% |
| frequency | 50 | 50 | 100.0% |
| importance | 50 | 50 | 100.0% |
| contradiction | 30 | 50 | 60.0% |

## Category Descriptions

- **recency_override**: Newer information should override older (e.g., job changes)
- **frequency**: Frequently mentioned items should rank higher (e.g., favorite foods)
- **importance**: High-importance memories should persist despite age (e.g., allergies)
- **contradiction**: Direct contradictions where latest state wins (e.g., relationship status)

## Analysis

- **Best category**: frequency (100.0%)
- **Worst category**: recency_override (60.0%)
