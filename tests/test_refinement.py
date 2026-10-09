"""Temporal registration and hard-negative correctness for scorer training."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from graft.catalog import Catalog, INF
from graft.config import SearchConfig
from graft.io import parameter_hash
from graft.refinement import CompetitionObjective, RefinementConfig, unique_visible_negatives
from graft.registration import register_arrivals
from graft.scorer import PrefixConditioner
from graft.training_episodes import episode_support


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.vectors = F.normalize(torch.randn(9, 8), dim=-1)
        self.vectors[0].zero_()
        books = [np.random.default_rng(10 + k).normal(size=(4, 8)).astype(np.float32) for k in range(3)]
        codes = np.full((9, 3), -1, np.int32)
        codes[1:7] = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 2]]
        first = np.array([INF, 1, 2, 3, 4, 5, 20, 25, 30], np.int64)
        self.base = Catalog(books, codes, first, np.zeros(4, np.int64), (4, 4, 4), [], codes[:, 0] >= 0)

    def test_negative_priority_visibility_and_duplicates(self):
        candidates = torch.tensor([[2, 3, 2, 0, 4, 1], [1, 3, 2, 3, 4, 0]])
        targets = torch.tensor([0, 2])
        birth = torch.tensor([0, 2, 3, 9, 5])
        chosen = unique_visible_negatives(candidates, targets, birth, torch.tensor([6, 10]), 4)
        torch.testing.assert_close(chosen, torch.tensor([[2, 4, 1, -1], [1, 3, 4, 0]]))

    def test_mining_includes_sibling_excludes_positive_and_future(self):
        search = SearchConfig(coarse_beam=4, leaf_beam=4)
        objective = CompetitionObjective(self.base, self.base, self.vectors, "cpu", search)
        config = RefinementConfig(negatives=4, sibling_budget=3, routed_budget=2)
        targets = torch.tensor([1, 4])
        positive, negative, _, _, stats = objective.mine(self.vectors[targets], targets, torch.tensor([5, 5]), config)
        self.assertTrue(bool((negative[:, 0] == objective.target_leaves(torch.tensor([2, 5]))).all()))
        self.assertTrue(bool((negative != positive[:, None]).all()))
        self.assertTrue(bool((objective.leaf_birth[negative.clamp(min=0)] <= 5).all()))
        for row in negative.tolist():
            valid = [x for x in row if x >= 0]
            self.assertEqual(len(valid), len(set(valid)))
        self.assertGreaterEqual(float(stats["sibling_negatives"]), 1)

    def test_support_and_arrivals_preserve_deployment_invariants(self):
        support = episode_support(self.base, 5)
        self.assertTrue(bool((support.codes[6:] == -1).all()))
        # Later base-registered content is removed from support descriptors/members.
        self.assertFalse(support.training_seen[6])
        self.assertEqual(support.first[7], INF)
        grown = register_arrivals(support, self.vectors.numpy(), 20, np.arange(9), 2)
        old = support.codes[:, 0] >= 0
        np.testing.assert_array_equal(grown.codes[old], support.codes[old])
        for before, after in zip(support.books, grown.books):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(len(grown.append_trace), 1)
        self.assertTrue(grown.append_trace[0]["fresh_path_admitted"])
        self.assertEqual(grown.append_trace[0]["item_id"], 6)
        self.assertTrue(bool((grown.codes[7:] == -1).all()))

    def test_loss_updates_only_shared_parameters_with_finite_gradients(self):
        support = episode_support(self.base, 5)
        grown = register_arrivals(support, self.vectors.numpy(), 20, np.arange(9), 2)
        objective = CompetitionObjective(support, grown, self.vectors, "cpu", SearchConfig(coarse_beam=4))
        config = RefinementConfig(negatives=4, sibling_budget=3, routed_budget=2)
        model = PrefixConditioner(8, 12)
        before = parameter_hash(model)
        vectors_before = self.vectors.clone()
        targets = torch.tensor([1, 4, 6])
        loss, stats = objective(model, self.vectors[targets], targets, torch.tensor([5, 5, 20]), config)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer.step()
        self.assertNotEqual(before, parameter_hash(model))
        torch.testing.assert_close(self.vectors, vectors_before, rtol=0, atol=0)
        self.assertIsNone(self.vectors.grad)
        self.assertEqual(
            set(stats), {"sibling_negatives", "valid_negatives", "reachable_targets", "path", "anchor", "teacher"}
        )

    def test_frozen_mining_prefix_scores_match_actual_beam_search(self):
        search = SearchConfig(coarse_beam=2, leaf_beam=4)
        objective = CompetitionObjective(self.base, self.base, self.vectors, "cpu", search)
        query = self.vectors[torch.tensor([1, 4])]
        times = torch.tensor([5, 5])
        expected_scores, retained = objective.prefix_scores(query, times)
        _, _, _, stages = objective.retriever.recommend(query, times, search)
        paths, scores = stages[1]["paths"], stages[1]["scores"]
        keys = paths[..., 0] * objective.sizes[1] + paths[..., 1]
        positions = torch.searchsorted(objective.pair_keys, keys)
        torch.testing.assert_close(expected_scores.gather(1, positions), scores)
        self.assertTrue(bool(retained.gather(1, positions).all()))

    def test_frozen_teacher_and_new_leaf_supervision_have_finite_gradients(self):
        support = episode_support(self.base, 5)
        grown = register_arrivals(support, self.vectors.numpy(), 20, np.arange(9), 2)
        objective = CompetitionObjective(support, grown, self.vectors, "cpu", SearchConfig(coarse_beam=4))
        model = PrefixConditioner(8, 12)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, 0.02)
        config = RefinementConfig(
            negatives=8, sibling_budget=3, routed_budget=2, teacher_weight=10, episode_new_only=True
        )
        targets = torch.tensor([1, 4, 6])
        loss, stats = objective(model, self.vectors[targets], targets, torch.tensor([5, 5, 20]), config)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(stats["teacher"].detach()), -1e-6)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_unreachable_target_produces_no_leaf_supervision(self):
        search = SearchConfig(coarse_beam=1, leaf_beam=2)
        objective = CompetitionObjective(self.base, self.base, self.vectors, "cpu", search)
        config = RefinementConfig(negatives=4, sibling_budget=3, routed_budget=2, anchor_weight=0, reachable_only=True)
        query = self.vectors[1:2]
        times = torch.tensor([5])
        _, retained = objective.prefix_scores(query, times)
        targets = [
            i
            for i in range(1, 6)
            if not retained[0, objective.leaf_pair[objective.target_leaves(torch.tensor([i]))[0]]]
        ]
        self.assertTrue(targets)
        model = PrefixConditioner(8, 12)
        loss, terms = objective(model, query, torch.tensor(targets[:1]), times, config)
        self.assertEqual(float(terms["reachable_targets"]), 0)
        self.assertEqual(float(terms["path"].detach()), 0)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_trained_teacher_stays_frozen_while_student_refines(self):
        import copy

        teacher = PrefixConditioner(8, 12)
        with torch.no_grad():
            teacher.net[-1].weight.normal_(0, 0.02)
        teacher.eval().requires_grad_(False)
        teacher_hash = parameter_hash(teacher)
        student = copy.deepcopy(teacher).requires_grad_(True)
        support = episode_support(self.base, 5)
        grown = register_arrivals(support, self.vectors.numpy(), 20, np.arange(9), 2)
        objective = CompetitionObjective(support, grown, self.vectors, "cpu", SearchConfig(coarse_beam=4), teacher)
        targets = torch.tensor([1, 4, 6])
        loss, _ = objective(
            student,
            self.vectors[targets],
            targets,
            torch.tensor([5, 5, 20]),
            RefinementConfig(negatives=8, teacher_weight=100),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        torch.optim.AdamW(student.parameters(), lr=0.001).step()
        self.assertEqual(parameter_hash(teacher), teacher_hash)
        self.assertNotEqual(parameter_hash(student), teacher_hash)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
