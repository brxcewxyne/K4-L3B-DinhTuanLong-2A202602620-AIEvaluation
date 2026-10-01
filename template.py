"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    # TODO: define fields
    # Hints:
    #   context: str = ""
    #   metadata: dict = field(default_factory=dict)
    #   retrieved_contexts: list = field(default_factory=list)
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    # TODO: define fields
    # Hints:
    #   failure_type: str | None = None
    #   context_precision: float | None = None
    #   context_recall: float | None = None
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        TODO: Return mean of the three metric scores
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context)
        score = len(answer_tokens & context_tokens) / len(answer_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        score = len(answer_tokens & question_tokens) / len(question_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        score = len(answer_tokens & expected_tokens) / len(expected_tokens)
        return max(0.0, min(1.0, float(score)))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in (contexts or []):
            union_tokens |= _tokenize(chunk)
        score = len(expected_tokens & union_tokens) / len(expected_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0
        relevant_flags: list[bool] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            coverage = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            relevant_flags.append(coverage >= relevance_threshold)
        total_relevant = sum(1 for flag in relevant_flags if flag)
        if total_relevant == 0:
            return 0.0
        running_relevant = 0
        precision_sum = 0.0
        for rank, is_relevant in enumerate(relevant_flags, start=1):
            if is_relevant:
                running_relevant += 1
                precision_sum += running_relevant / rank
        score = precision_sum / total_relevant
        return max(0.0, min(1.0, float(score)))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        passed = (
            faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5
        )
        if faithfulness < 0.3:
            failure_type: str | None = "hallucination"
        elif relevance < 0.3:
            failure_type = "irrelevant"
        elif completeness < 0.3:
            failure_type = "incomplete"
        elif not passed:
            failure_type = "off_topic"
        else:
            failure_type = None

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=list(contexts) if contexts is not None else [],
        )
        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    # TODO (Bonus — Exercise 3.5): implement the reranker
    raise NotImplementedError("Implement rerank_by_overlap")


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        rubric_lines = "\n".join(
            f"- {name}: {desc} (score 0-1)"
            for name, desc in rubric.items()
        )
        prompt = (
            "You are an impartial evaluator for OrbitTech Store customer support.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Rubric (criterion: description):\n"
            f"{rubric_lines}\n"
            "Instructions: Score EACH rubric criterion on a 0-1 scale "
            "(0 = completely wrong, 1 = perfect). "
            "Return ONLY valid JSON mapping each criterion name to its numeric "
            'score, e.g. {"accuracy": 0.8, "clarity": 0.7}. '
            "Do not use the 1-5 scale; use 0-1."
        )
        raw = self.judge_llm_fn(prompt)
        reasoning = raw if isinstance(raw, str) else str(raw)

        def _coerce(value: Any) -> float:
            # Deterministic fallback for missing/invalid/out-of-range scores.
            if isinstance(value, bool):
                return 0.5
            try:
                number = float(value) if not isinstance(value, float) else value
            except (TypeError, ValueError):
                return 0.5
            if not (0.0 <= number <= 1.0) or number != number:
                return 0.5
            return float(number)

        try:
            text = raw.strip() if isinstance(raw, str) else str(raw)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found")
                parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not an object")
            if isinstance(parsed.get("scores"), dict):
                score_map = parsed["scores"]
                nested_reasoning = parsed.get("reasoning")
                if isinstance(nested_reasoning, str) and nested_reasoning.strip():
                    reasoning = nested_reasoning
            else:
                score_map = parsed
                if isinstance(score_map.get("reasoning"), str):
                    reasoning = score_map["reasoning"]
            scores = {
                criterion: _coerce(score_map.get(criterion, 0.5))
                for criterion in rubric
            }
        except Exception:
            scores = {criterion: 0.5 for criterion in rubric}
        return {"scores": scores, "reasoning": reasoning}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        # Note: ordinary {"scores": {...}} batches carry no response-position
        # information, so positional bias cannot be inferred from them alone.
        # Only explicit positional evidence (e.g. first/second score pairs or
        # a preferred-position field) may set positional_bias=True; otherwise
        # it stays False to avoid fabricating observations.
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }
        numeric_scores: list[float] = []
        for entry in scores_batch:
            if not isinstance(entry, dict):
                continue
            score_map = entry.get("scores")
            if not isinstance(score_map, dict):
                continue
            for value in score_map.values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    number = float(value)
                    if number != number:
                        continue
                    if number in (float("inf"), float("-inf")):
                        continue
                    numeric_scores.append(number)
        if numeric_scores:
            average = sum(numeric_scores) / len(numeric_scores)
            leniency_bias = average > 0.8
            severity_bias = average < 0.3
        else:
            leniency_bias = False
            severity_bias = False

        positional_bias = False
        pairs: list[tuple[float, float]] = []
        for entry in scores_batch:
            if not isinstance(entry, dict):
                continue
            try:
                if "first_score" in entry and "second_score" in entry:
                    first = float(entry["first_score"])
                    second = float(entry["second_score"])
                    if isinstance(entry["first_score"], bool) or isinstance(
                        entry["second_score"], bool
                    ):
                        continue
                    pairs.append((first, second))
                elif "scores_first" in entry and "scores_second" in entry:
                    first_map = entry["scores_first"]
                    second_map = entry["scores_second"]
                    if isinstance(first_map, dict) and isinstance(
                        second_map, dict
                    ):
                        first_vals = [
                            float(v)
                            for v in first_map.values()
                            if isinstance(v, (int, float))
                            and not isinstance(v, bool)
                        ]
                        second_vals = [
                            float(v)
                            for v in second_map.values()
                            if isinstance(v, (int, float))
                            and not isinstance(v, bool)
                        ]
                        if first_vals and second_vals:
                            pairs.append(
                                (
                                    sum(first_vals) / len(first_vals),
                                    sum(second_vals) / len(second_vals),
                                )
                            )
                elif "preferred_position" in entry:
                    preferred = str(entry["preferred_position"]).strip().lower()
                    if preferred in ("first", "a", "1"):
                        pairs.append((1.0, 0.0))
                    elif preferred in ("second", "b", "2"):
                        pairs.append((0.0, 1.0))
            except (TypeError, ValueError):
                continue
        if pairs:
            first_wins = sum(1 for first, second in pairs if first > second)
            positional_bias = first_wins > len(pairs) / 2

        return {
            "positional_bias": bool(positional_bias),
            "leniency_bias": bool(leniency_bias),
            "severity_bias": bool(severity_bias),
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts,
            )
            # Preserve the ORIGINAL QAPair instance (identity, order,
            # metadata.id, retrieved_contexts) even if the evaluator built
            # a new QAPair internally.
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        pass_rate = (passed / total) if total else 0.0
        if total:
            avg_faithfulness = sum(r.faithfulness for r in results) / total
            avg_relevance = sum(r.relevance for r in results) / total
            avg_completeness = sum(r.completeness for r in results) / total
        else:
            avg_faithfulness = 0.0
            avg_relevance = 0.0
            avg_completeness = 0.0
        recall_values = [
            r.context_recall for r in results if r.context_recall is not None
        ]
        precision_values = [
            r.context_precision
            for r in results
            if r.context_precision is not None
        ]
        avg_context_recall: float | None = (
            sum(recall_values) / len(recall_values) if recall_values else None
        )
        avg_context_precision: float | None = (
            sum(precision_values) / len(precision_values)
            if precision_values
            else None
        )
        failure_types: dict[str, int] = {}
        for result in results:
            if not result.passed and result.failure_type is not None:
                failure_types[result.failure_type] = (
                    failure_types.get(result.failure_type, 0) + 1
                )
        return {
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_context_recall": avg_context_recall,
            "avg_context_precision": avg_context_precision,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions

        TODO: Compute avg per metric, compare, list regressions, set passed flag
        """
        def _mean(values: list[float]) -> float:
            # No-data behavior: an empty run averages to 0.0 and yields no
            # regressions (passed=True), to avoid inventing evidence of a
            # drop when there is nothing to compare.
            return sum(values) / len(values) if values else 0.0

        new_faith = _mean([r.faithfulness for r in new_results])
        new_rel = _mean([r.relevance for r in new_results])
        new_comp = _mean([r.completeness for r in new_results])
        base_faith = _mean([r.faithfulness for r in baseline_results])
        base_rel = _mean([r.relevance for r in baseline_results])
        base_comp = _mean([r.completeness for r in baseline_results])

        regressions: list[str] = []

        def _regressed(baseline: float, new: float) -> bool:
            drop = baseline - new
            # Exactly 0.05 is NOT a regression; tolerance guards float noise
            # (e.g. 0.9 - 0.85 == 0.050000000000000044) without hiding a
            # genuine drop such as 0.06.
            if math.isclose(drop, 0.05, abs_tol=1e-9):
                return False
            return drop > 0.05

        if _regressed(base_faith, new_faith):
            regressions.append("faithfulness")
        if _regressed(base_rel, new_rel):
            regressions.append("relevance")
        if _regressed(base_comp, new_comp):
            regressions.append("completeness")
        # Empty either side: _mean gives 0.0 for the empty side; to avoid
        # inventing a regression from missing data, report no regressions.
        if not new_results or not baseline_results:
            regressions = []
        return {
            "new_avg_faithfulness": new_faith,
            "new_avg_relevance": new_rel,
            "new_avg_completeness": new_comp,
            "baseline_avg_faithfulness": base_faith,
            "baseline_avg_relevance": base_rel,
            "baseline_avg_completeness": base_comp,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            result
            for result in results
            if result.faithfulness < threshold
            or result.relevance < threshold
            or result.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        counts: dict[str, int] = {}
        for failure in failures:
            failure_type = getattr(failure, "failure_type", None)
            if failure_type is None:
                continue
            counts[failure_type] = counts.get(failure_type, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        # Heuristic hypothesis only: score comparison alone cannot prove a
        # retrieval vs generation root cause. Confirm with gold evidence,
        # retrieved chunks and execution traces before claiming true RCA.
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        minimum = min(scores.values())
        lowest = [name for name, value in scores.items() if value == minimum]
        if len(lowest) > 1:
            return "Multiple issues detected — review full pipeline"
        if lowest[0] == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        if lowest[0] == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        TODO: Build markdown table with failure details + matched suggestions
        """
        def _escape(cell: Any) -> str:
            text = str(cell) if cell is not None else "None"
            return text.replace("|", "\\|").replace("\n", " ").strip()

        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for index, failure in enumerate(failures, start=1):
            failure_id = f"F{index:03d}"
            failure_type = _escape(failure.failure_type)
            root_cause = _escape(self.find_root_cause(failure))
            if index - 1 < len(suggestions):
                fix = _escape(suggestions[index - 1])
            else:
                fix = "TBD — assign owner and verify with relevant metric"
            lines.append(
                f"| {failure_id} | {failure_type} | {root_cause} | {fix} | Open |"
            )
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []
        # Diagnostic hypotheses below are prioritized cross-case fixes, not
        # proven root causes; each names the metric/trace used to verify it.
        counts = self.categorize_failures(failures)
        lowered = {str(key).lower(): value for key, value in counts.items()}
        suggestions: list[str] = []
        if lowered.get("hallucination"):
            suggestions.append(
                "Implement a hallucination checker requiring a corpus citation "
                "for every policy claim (return windows, restocking fees, "
                "warranty periods); verify with faithfulness and Context "
                "Precision traces before answering"
            )
        if lowered.get("irrelevant") or lowered.get("off_topic"):
            suggestions.append(
                "Clarify intent routing in the system prompt (return vs "
                "warranty vs repair vs membership) with few-shot routing "
                "examples; verify with Answer Relevance on ambiguous queries"
            )
        if lowered.get("incomplete") or lowered.get("low_completeness"):
            suggestions.append(
                "Add an answer checklist covering mandatory terms, conditions "
                "and exceptions (restocking-fee waiver for verified defects, "
                "proof of purchase, refund timing); verify with completeness "
                "and Context Recall"
            )
        suggestions.append(
            "Improve retriever chunking and query rewriting and rerank "
            "policy-condition chunks to the top; verify with Context Recall "
            "and rank-aware Context Precision"
        )
        suggestions.append(
            "Tune refusal guardrails to the scope policy so in-scope "
            "OrbitTech questions are answered while prompt-injection and "
            "out-of-scope requests are safely declined; verify with human "
            "review on adversarial cases"
        )
        suggestions.append(
            "Augment the golden dataset with the observed failure patterns "
            "and enforce a CI regression gate blocking deploys on metric "
            "drops > 0.05; verify with BenchmarkRunner.run_regression"
        )
        # Deduplicate while keeping deterministic order, then guarantee >= 3.
        seen: set[str] = set()
        unique: list[str] = []
        for suggestion in suggestions:
            if suggestion and suggestion not in seen:
                seen.add(suggestion)
                unique.append(suggestion)
        while len(unique) < 3:
            unique.append(
                "Expand context window for multi-condition policy answers and "
                "re-check completeness against gold evidence"
            )
        return unique


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
