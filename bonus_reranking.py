"""Bonus Exercise 3.5 — offline lexical reranker (standalone, no API calls).

Method: candidate-set IDF-weighted question-term overlap.

Definitions:
    Q       = token set of the QUESTION (never the gold expected answer).
    C_i     = token set of candidate chunk i.
    N       = number of fixed candidate chunks for one QA case.
    df(t)   = number of candidate chunks containing term t (within this case only).
    idf(t)  = ln((N + 1) / (df(t) + 1)) + 1.
    score(Q, C_i) = sum(idf(t) for t in Q ∩ C_i).

Chunks are ordered by score, highest first. Ties keep the original BM25
order (stable sort). The tie-break carries no relevance signal; it only
makes the output deterministic.

How this differs from the BM25 retriever in domain_assistant.py:
    - No term-frequency saturation and no length normalization.
    - Binary term presence per chunk (a term counts once per chunk).
    - IDF is computed on the tiny fixed candidate set (N = 5), not the corpus.
    - Input is the user QUESTION only, like a query rewrite at rerank time.

Fairness controls (enforced by run_experiment and the bonus tests):
    - Same chunk IDs before and after (verified per case).
    - No chunk added, deleted, replaced, or reworded.
    - No second retrieval; no gold expected_answer, gold contexts, labels,
      or benchmark scores are visible to rerank().
    - Gold evidence is used ONLY by the official Context Precision/Recall
      measurement (template.RAGASEvaluator), never by the reranker.

Limitations:
    - Purely lexical: no stemming and no synonyms, so pairs such as
      deliver/delivery or day/date still mismatch.
    - IDF on N = 5 is unstable; a term in 1 of 5 chunks dominates the score.
    - Reordering cannot recover gold evidence absent from the candidate set
      (e.g. the missing OT-08-P02 paragraph for M06).
    - Short or stopword-heavy questions produce many ties (order unchanged).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopword list for the reranker. Documented here so the
# experiment is reproducible without importing evaluation internals.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
    "i", "you", "my", "we", "what", "how", "why", "when", "which", "who",
    "can", "could", "should", "would", "do", "does", "did", "will", "if",
    "not", "no", "me", "your",
})

EXPERIMENT_VERSION = "bonus-3.5-idf-rerank-v1"

# Case selection (documented BEFORE running; non-score-based eligibility):
# difficulty/failure-pattern variation across Easy/Medium/Hard/Adversarial,
# preferring cases with a full 5-chunk candidate set. A01 is excluded by the
# eligibility rule (>= 3 chunks required for a meaningful reorder; A01 has 2).
SELECTED_IDS = ["E01", "M01", "M04", "M06", "H04", "A03"]


def tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens minus stopwords (set, order-free)."""
    if not text:
        return set()
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def idf_weights(chunk_token_sets: list[set[str]]) -> dict[str, float]:
    """IDF over the fixed candidate set only. No corpus, no gold labels."""
    n = len(chunk_token_sets)
    df: dict[str, int] = {}
    for tokens in chunk_token_sets:
        for term in tokens:
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def score_chunk(
    question_tokens: set[str],
    chunk_tokens: set[str],
    idf: dict[str, float],
) -> float:
    """Sum of candidate-set IDF weights of shared question terms."""
    return sum(idf.get(term, 0.0) for term in question_tokens & chunk_tokens)


def rerank(question: str, chunks: list) -> list:
    """Reorder chunk items by question-term IDF overlap (stable on ties).

    Args:
        question: the original user question (NOT the gold expected answer).
        chunks: list of chunk dicts (each with a "text" key) or plain
            strings. The SAME objects are returned, only reordered.

    Returns:
        New list with the same items in reranked order.
    """
    token_sets: list[set[str]] = []
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        token_sets.append(tokenize(text))
    idf = idf_weights(token_sets)
    question_tokens = tokenize(question)
    scored = [
        (score_chunk(question_tokens, tokens, idf), index)
        for index, tokens in enumerate(token_sets)
    ]
    # Highest score first; ties keep original BM25 order (stable, documented).
    order = sorted(range(len(chunks)), key=lambda i: (-scored[i][0], i))
    return [chunks[i] for i in order]


def _chunk_id(chunk: dict) -> str:
    return str(chunk.get("chunk_id", chunk.get("text", ""))[:120])


def run_experiment(
    golden_path: str | Path = "golden_dataset.json",
    actual_path: str | Path = "artifacts/actual_answers.json",
    output_path: str | Path = "artifacts/bonus_reranking_results.json",
) -> dict:
    """Run the fixed-candidate rerank experiment and persist measurements."""
    from template import RAGASEvaluator  # official metric only, not reranking

    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    actual = json.loads(Path(actual_path).read_text(encoding="utf-8"))
    expected_by_id = {p["id"]: p["expected_answer"] for p in golden["qa_pairs"]}
    question_by_id = {p["id"]: p["question"] for p in golden["qa_pairs"]}
    actual_by_id = {a["id"]: a for a in actual["answers"]}
    evaluator = RAGASEvaluator()

    cases = []
    for qid in SELECTED_IDS:
        retrieved = actual_by_id[qid]["retrieved_contexts"]
        baseline_ids = [_chunk_id(c) for c in retrieved]
        baseline_texts = [c["text"] for c in retrieved]
        expected = expected_by_id[qid]
        recall_before = evaluator.evaluate_context_recall(baseline_texts, expected)
        precision_before = evaluator.evaluate_context_precision(
            baseline_texts, expected
        )
        reranked = rerank(question_by_id[qid], retrieved)
        reranked_ids = [_chunk_id(c) for c in reranked]
        reranked_texts = [c["text"] for c in reranked]
        assert sorted(baseline_ids) == sorted(reranked_ids), (
            f"candidate set changed for {qid}"
        )
        recall_after = evaluator.evaluate_context_recall(reranked_texts, expected)
        precision_after = evaluator.evaluate_context_precision(
            reranked_texts, expected
        )
        cases.append({
            "id": qid,
            "question": question_by_id[qid],
            "baseline_chunk_ids": baseline_ids,
            "reranked_chunk_ids": reranked_ids,
            "candidate_set_identical": (
                sorted(baseline_ids) == sorted(reranked_ids)
            ),
            "recall_before": recall_before,
            "recall_after": recall_after,
            "precision_before": precision_before,
            "precision_after": precision_after,
            "delta_precision": precision_after - precision_before,
        })

    n = len(cases)
    result = {
        "experiment": EXPERIMENT_VERSION,
        "method": (
            "candidate-set IDF-weighted question-term overlap; "
            "score = sum idf(t) over shared question terms; stable ties"
        ),
        "reranker_inputs": "question + candidate chunk texts only (no gold)",
        "metric": "template.RAGASEvaluator.evaluate_context_precision/recall",
        "cases": cases,
        "mean_recall_before": sum(c["recall_before"] for c in cases) / n,
        "mean_recall_after": sum(c["recall_after"] for c in cases) / n,
        "mean_precision_before": sum(
            c["precision_before"] for c in cases
        ) / n,
        "mean_precision_after": sum(c["precision_after"] for c in cases) / n,
        "mean_delta_precision": sum(
            c["delta_precision"] for c in cases
        ) / n,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    summary = run_experiment()
    for case in summary["cases"]:
        print(
            "%s recall %.3f -> %.3f | precision %.3f -> %.3f (delta %+.3f) | set_ok=%s"
            % (
                case["id"],
                case["recall_before"],
                case["recall_after"],
                case["precision_before"],
                case["precision_after"],
                case["delta_precision"],
                case["candidate_set_identical"],
            )
        )
    print(
        "mean precision %.3f -> %.3f (delta %+.3f)" % (
            summary["mean_precision_before"],
            summary["mean_precision_after"],
            summary["mean_delta_precision"],
        )
    )
    print("mean recall %.3f -> %.3f" % (
        summary["mean_recall_before"], summary["mean_recall_after"]))
