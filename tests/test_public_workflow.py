"""Protocol checks for split separation, seed statistics and external predictions."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from graft.aggregation import aggregate
from graft.config import load_config
from graft.data import right_pad
from graft.external_predictions import evaluate_predictions
from graft.preprocessing import prepare_events


class PublicWorkflowTests(unittest.TestCase):
    def test_equal_time_histories_and_train_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [("u", "a", 1), ("u", "b", 1), ("u", "c", 2), ("u", "d", 3), ("u", "a", 4), ("u", "e", 5)]
            with (root / "events.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["user_id", "item_id", "timestamp"])
                writer.writerows(events)
            np.savez(root / "features.npz", ids=np.array(list("abcde")), vectors=np.eye(5, dtype=np.float32))
            report = prepare_events(root / "events.csv", root / "features.npz", 2, 4, root / "data", max_history=4)
            self.assertEqual(report["split_audit"]["train"]["p2_nonempty_rows"], 1)
            mapping = json.loads((root / "data/item_mapping.json").read_text())
            with np.load(root / "data/p2_train.npz") as train:
                self.assertEqual(train["timestamp"].tolist(), [2])
                self.assertEqual(set(train["history"][0]) - {0}, {mapping["a"], mapping["b"]})
            with np.load(root / "data/catalog.npz") as catalog:
                self.assertTrue(catalog["training_seen"][mapping["a"]])
                self.assertFalse(catalog["training_seen"][mapping["d"]])
            with np.load(root / "data/p2_valid.npz") as valid, np.load(root / "data/p2_test.npz") as test:
                self.assertEqual(valid["timestamp"].tolist(), [3, 4])
                self.assertEqual(test["timestamp"].tolist(), [5])

    def test_padding_preserves_sequence_order(self):
        history, length = right_pad(np.array([[0, 0, 2, 3], [0, 4, 5, 6]]))
        np.testing.assert_array_equal(history, [[2, 3, 0, 0], [4, 5, 6, 0]])
        np.testing.assert_array_equal(length, [2, 3])

    def test_aggregate_calculates_sample_sd_and_rejects_cohort_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed, score in zip((10, 11, 12), (0.1, 0.2, 0.3)):
                row = dict(
                    dataset="Games",
                    method="Graft",
                    split="test",
                    seed=seed,
                    full_split=True,
                    search={"new_leaf_bias": 1},
                    identity_query=False,
                    parent_descriptor=False,
                    source_hashes={"queries": "same-query-arrays"},
                    metrics={
                        group: {"count": 100, "recall_50": score, "ndcg_50": score / 2} for group in ("overall", "cold")
                    },
                )
                path = root / f"seed_{seed}.json"
                path.write_text(json.dumps(row))
                paths.append(path)
            summary = aggregate(paths, root / "summary.json")
            self.assertAlmostEqual(summary["metrics"]["cold"]["recall_50"]["mean"], 0.2)
            self.assertAlmostEqual(summary["metrics"]["cold"]["recall_50"]["standard_deviation"], 0.1)
            changed = json.loads(paths[2].read_text())
            changed["source_hashes"]["queries"] = "different-arrays"
            paths[2].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "different query"):
                aggregate(paths, root / "invalid.json")

    def test_external_predictions_reject_future_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/Video_Games"
            data.mkdir(parents=True)
            np.savez(data / "p2_test.npz", item=[1], timestamp=[10])
            np.savez(data / "catalog.npz", first_timestamp=[999, 1, 20], training_seen=[False, True, False])
            output = np.zeros((1, 50), np.int32)
            output[0, :2] = [1, 2]
            np.savez(root / "baseline.npz", items=output, query_index=[0])
            config = load_config("Games", workspace=root)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                evaluate_predictions(config, root / "baseline.npz", root / "result", "Baseline")


if __name__ == "__main__":
    unittest.main()
