from __future__ import annotations

import unittest

from um_agente_de_ia.evaluation import (
    context_recall,
    exact_match,
    f1_score,
    mean_reciprocal_rank,
    precision_at_k,
)
from um_agente_de_ia.training import TrainingExample, augment_dataset, split_dataset


class TrainingAndEvaluationTests(unittest.TestCase):
    def test_split_dataset(self):
        dataset = [
            TrainingExample("q1", "a1"),
            TrainingExample("q2", "a2"),
            TrainingExample("q3", "a3"),
            TrainingExample("q4", "a4"),
        ]
        train, test = split_dataset(dataset, train_ratio=0.5)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)

    def test_split_dataset_accepts_bounds(self):
        dataset = [TrainingExample("q1", "a1"), TrainingExample("q2", "a2")]
        train_none, test_all = split_dataset(dataset, train_ratio=0)
        train_all, test_none = split_dataset(dataset, train_ratio=1)
        self.assertEqual(len(train_none), 0)
        self.assertEqual(len(test_all), 2)
        self.assertEqual(len(train_all), 2)
        self.assertEqual(len(test_none), 0)

    def test_split_dataset_shuffle_reproducible(self):
        dataset = [TrainingExample(f"q{i}", f"a{i}") for i in range(10)]
        train1, _ = split_dataset(dataset, shuffle=True, random_seed=42)
        train2, _ = split_dataset(dataset, shuffle=True, random_seed=42)
        self.assertEqual([e.question for e in train1], [e.question for e in train2])

    def test_split_dataset_shuffle_different_seed(self):
        dataset = [TrainingExample(f"q{i}", f"a{i}") for i in range(20)]
        train1, _ = split_dataset(dataset, shuffle=True, random_seed=1)
        train2, _ = split_dataset(dataset, shuffle=True, random_seed=2)
        # Very unlikely to be identical with different seeds
        self.assertNotEqual([e.question for e in train1], [e.question for e in train2])

    def test_exact_match_and_context_recall(self):
        self.assertEqual(exact_match("Resposta", "resposta"), 1.0)
        self.assertEqual(context_recall(["A", "B"], ["B", "C"]), 0.5)

    def test_precision_at_k(self):
        relevant = ["A", "B", "C"]
        retrieved = ["A", "X", "B", "Y"]
        self.assertAlmostEqual(precision_at_k(relevant, retrieved, k=2), 0.5)
        self.assertAlmostEqual(precision_at_k(relevant, retrieved, k=4), 0.5)
        self.assertEqual(precision_at_k(relevant, retrieved, k=0), 0.0)
        self.assertEqual(precision_at_k(relevant, [], k=3), 0.0)

    def test_f1_score(self):
        relevant = ["A", "B"]
        retrieved = ["A", "C"]
        recall = context_recall(relevant, retrieved)
        precision = precision_at_k(relevant, retrieved, k=len(retrieved))
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(f1_score(relevant, retrieved), expected)

    def test_f1_score_perfect(self):
        self.assertAlmostEqual(f1_score(["A", "B"], ["A", "B"]), 1.0)

    def test_f1_score_zero(self):
        self.assertAlmostEqual(f1_score(["A"], ["B"]), 0.0)

    def test_f1_score_empty(self):
        self.assertAlmostEqual(f1_score([], []), 1.0)

    def test_mean_reciprocal_rank_first(self):
        self.assertAlmostEqual(mean_reciprocal_rank(["A"], ["A", "B", "C"]), 1.0)

    def test_mean_reciprocal_rank_second(self):
        self.assertAlmostEqual(mean_reciprocal_rank(["B"], ["A", "B", "C"]), 0.5)

    def test_mean_reciprocal_rank_no_hit(self):
        self.assertAlmostEqual(mean_reciprocal_rank(["Z"], ["A", "B", "C"]), 0.0)

    def test_augment_dataset_produces_variations(self):
        dataset = [TrainingExample("Há um problema crítico no sistema", "Investigar imediatamente")]
        augmented = augment_dataset(dataset, random_seed=0)
        # original + at least one variation
        self.assertGreater(len(augmented), 1)
        questions = [e.question for e in augmented]
        self.assertIn("Há um problema crítico no sistema", questions)

    def test_augment_dataset_preserves_all_originals(self):
        dataset = [
            TrainingExample("q1 sem palavras especiais", "a1"),
            TrainingExample("problema grave q2", "a2"),
        ]
        augmented = augment_dataset(dataset, random_seed=0)
        originals = {e.question for e in dataset}
        for original in originals:
            self.assertIn(original, [e.question for e in augmented])

    def test_augment_dataset_custom_synonym_map(self):
        dataset = [TrainingExample("O banco está lento", "Reiniciar")]
        custom_map = {"banco": ["database", "BD"]}
        augmented = augment_dataset(dataset, synonym_map=custom_map, random_seed=0)
        questions = [e.question for e in augmented]
        self.assertTrue(any("database" in q or "BD" in q for q in questions))


if __name__ == "__main__":
    unittest.main()
