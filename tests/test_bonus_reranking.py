"""Bonus Exercise 3.5 tests — reranker behavior only (no API, no gold leak).

These tests are separate from the official suite in tests/test_solution.py
and do not modify any official contract.
"""

import inspect
import sys
import unittest
from pathlib import Path

DAY_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DAY_DIR))

from bonus_reranking import rerank, score_chunk, tokenize  # noqa: E402
from template import RAGASEvaluator  # noqa: E402


def _chunks(*texts):
    return [
        {"chunk_id": f"C{i:02d}", "text": text}
        for i, text in enumerate(texts)
    ]


class TestRerankDeterminism(unittest.TestCase):
    def test_same_input_same_order(self):
        chunks = _chunks(
            "Paris is the capital of France",
            "Bananas are a tropical fruit",
            "France is located in Europe",
        )
        first = [c["chunk_id"] for c in rerank("capital of France", chunks)]
        second = [c["chunk_id"] for c in rerank("capital of France", chunks)]
        self.assertEqual(first, second)


class TestCandidateSetPreservation(unittest.TestCase):
    def test_same_ids_no_dup_no_loss(self):
        chunks = _chunks("alpha beta", "gamma delta", "beta gamma", "delta eps")
        out = rerank("beta gamma question", chunks)
        self.assertEqual(len(out), len(chunks))
        self.assertEqual(
            sorted(c["chunk_id"] for c in out),
            sorted(c["chunk_id"] for c in chunks),
        )

    def test_empty_and_single_chunk(self):
        self.assertEqual(rerank("anything", []), [])
        one = _chunks("only chunk here")
        self.assertEqual(rerank("only", one), one)


class TestTieBreaking(unittest.TestCase):
    def test_all_ties_keep_original_order(self):
        chunks = _chunks("zzz qq", "ww ee", "rr tt")
        out = rerank("completely unrelated words here", chunks)
        self.assertEqual(
            [c["chunk_id"] for c in out],
            [c["chunk_id"] for c in chunks],
        )


class TestNoGoldAccess(unittest.TestCase):
    def test_signature_has_no_gold_parameters(self):
        params = set(inspect.signature(rerank).parameters)
        self.assertEqual(params, {"question", "chunks"})
        for forbidden in ("expected", "gold", "label", "answer", "score"):
            self.assertNotIn(forbidden, params)

    def test_score_uses_question_terms_only(self):
        # Same chunks, different questions must be able to give different
        # orders, proving the expected answer plays no role.
        chunks = _chunks("Paris France capital", "banana tropical fruit")
        order_q1 = [c["chunk_id"] for c in rerank("capital France", chunks)]
        order_q2 = [c["chunk_id"] for c in rerank("banana fruit", chunks)]
        self.assertNotEqual(order_q1, order_q2)


class TestMetricContract(unittest.TestCase):
    def test_precision_definition_spot_check(self):
        evaluator = RAGASEvaluator()
        expected = "Paris is the capital of France"
        relevant = "Paris is the capital city of France"
        noise = "Bananas are yellow tropical fruits grown near the equator"
        self.assertAlmostEqual(
            evaluator.evaluate_context_precision([relevant, noise], expected),
            1.0,
        )
        self.assertAlmostEqual(
            evaluator.evaluate_context_precision([noise, relevant], expected),
            0.5,
        )

    def test_recall_invariant_to_order(self):
        evaluator = RAGASEvaluator()
        expected = "Paris is the capital of France"
        chunks = [
            "Paris is the capital",
            "France is located in Europe",
            "noise bananas",
        ]
        self.assertAlmostEqual(
            evaluator.evaluate_context_recall(chunks, expected),
            evaluator.evaluate_context_recall(list(reversed(chunks)), expected),
        )


if __name__ == "__main__":
    unittest.main()
