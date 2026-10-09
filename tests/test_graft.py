"""Regression tests against the frozen evaluator and deployment invariants."""

from __future__ import annotations
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from graft.catalog import Catalog
from graft.config import SearchConfig
from graft.evaluation import GraftRetriever
from graft.io import parameter_hash
from graft.registration import register_arrivals
from graft.scorer import PrefixConditioner
from graft.training import PathObjective, TrainingConfig


class GraftRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.vectors = F.normalize(torch.randn(8, 8), dim=-1)
        self.vectors[0].zero_()
        books = [np.random.default_rng(9 + level).normal(size=(3, 8)).astype(np.float32) for level in range(3)]
        codes = np.full((8, 3), -1, np.int32)
        codes[1:4] = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        first = np.array([np.iinfo(np.int64).max, 1, 2, 3, 10, 12, 20, 30], np.int64)
        seen = np.array([False, True, True, True, False, False, False, False])
        self.base = Catalog(books, codes, first, np.zeros(3, np.int64), (3, 3, 3), [], seen)
        self.grown = register_arrivals(self.base, self.vectors.numpy(), 20, np.arange(8), 2)
        self.scorer = PrefixConditioner(8, 12)
        with torch.no_grad():
            self.scorer.net[-1].weight.normal_(0, 0.03)

    def test_admission_freezes_books_old_addresses_and_prefix_support(self):
        old = self.base.codes[:, 0] >= 0
        np.testing.assert_array_equal(self.grown.codes[old], self.base.codes[old])
        for before, after in zip(self.base.books, self.grown.books):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(sum((row["fresh_path_admitted"] for row in self.grown.append_trace)), 2)
        prefixes = {tuple(code[:2]) for code in self.base.codes[old]}
        self.assertTrue(all((tuple(code[:2]) in prefixes for code in self.grown.codes[4:7])))
        np.testing.assert_array_equal(self.grown.codes[7], [-1, -1, -1])
        fixed = register_arrivals(self.base, self.vectors.numpy(), 20, np.arange(8), 0)
        old_paths = {tuple(code) for code in self.base.codes[old]}
        self.assertTrue(all((tuple(code) in old_paths for code in fixed.codes[4:7])))

    def test_future_admission_cannot_change_visible_recommendations(self):
        early = register_arrivals(self.base, self.vectors.numpy(), 10, np.arange(8), 2)
        changed = self.vectors.clone()
        changed[5:] = F.normalize(torch.randn_like(changed[5:]), dim=-1)
        before = GraftRetriever(self.base, early, self.vectors, self.scorer, "cpu")
        after = GraftRetriever(self.base, self.grown, changed, self.scorer, "cpu")
        search = SearchConfig(coarse_beam=20, leaf_beam=10, top_k=5)
        query = self.vectors[4:5]
        a = before.recommend(query, [10], search)[0]
        b = after.recommend(query, [10], search)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all((i == 0 or self.base.first[i] <= 10 for i in a[0].tolist())))

    def test_training_updates_shared_scorer_and_keeps_content_fixed(self):
        vectors = self.vectors.clone()
        before = vectors.clone()
        objective = PathObjective(self.base, vectors, "cpu")
        model = PrefixConditioner(8, 12)
        previous = parameter_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss, terms = objective(
            model, vectors[1:4], torch.tensor([1, 2, 3]), torch.tensor([3, 3, 3]), TrainingConfig(negatives=8)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"path", "graft", "anchor"})
        loss.backward()
        self.assertTrue(all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())))
        optimizer.step()
        self.assertNotEqual(previous, parameter_hash(model))
        torch.testing.assert_close(vectors, before, rtol=0, atol=0)
        self.assertIsNone(vectors.grad)


if __name__ == "__main__":
    unittest.main()
