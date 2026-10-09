"""A small deterministic CPU example covering the complete public workflow."""

import csv
import numpy as np

from .config import load_config
from .io import atomic_json, fresh_directory
from .preprocessing import prepare_events
from .workflows import pipeline


def create_demo(output, seed=2024, run=True):
    output = fresh_directory(output)
    generator = np.random.default_rng(19)
    ids = np.array([f"item_{i}" for i in range(40)])
    vectors = generator.normal(size=(40, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez_compressed(output / "features.npz", ids=ids, vectors=vectors)
    events = []
    for t in range(1, 721):
        if t <= 480:
            count = min(24, 4 + t // 20)
        elif t <= 600:
            count = 24 + min(8, 1 + (t - 481) // 15)
        else:
            count = 32 + min(8, 1 + (t - 601) // 15)
        # Expose each arriving item immediately; otherwise alternate old/new targets.
        item = count - 1 if t % 15 == 1 or t % 20 == 0 else int(generator.integers(count))
        events.append((f"user_{t % 8}", ids[item], t))
    with (output / "events.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "item_id", "timestamp"])
        writer.writerows(events)
    values = dict(
        dataset="Demo",
        domain="Demo",
        workspace=".",
        seed=seed,
        codebook_sizes=[4, 4, 8],
        leaf_budget=64,
        warmup_steps=0,
        scorer_checkpoint="artifacts/{dataset}/s{seed}/scorer/step_0004.pt",
        encoder=dict(
            max_seq_length=8,
            n_layers=1,
            n_heads=2,
            hidden_size=16,
            inner_size=32,
            adaptor_layers=[16, 16],
            n_exps=2,
            hidden_dropout_prob=0.1,
            attn_dropout_prob=0.1,
            adaptor_dropout_prob=0.1,
        ),
        encoder_training=dict(
            epochs=2,
            patience=2,
            lr=0.001,
            batch_size=64,
            eval_batch_size=32,
            eval_every=1,
            validation_size=32,
            selection_metric="ndcg_50",
            bf16=False,
        ),
        refinement=dict(
            steps=4,
            batch_size=16,
            episode_batch_size=16,
            negatives=8,
            sibling_budget=4,
            routed_budget=2,
            validation_size=16,
            checkpoints=[0, 4],
            base_path_weight=0,
            teacher_weight=100,
            episode_new_only=True,
            reachable_only=True,
        ),
        search=dict(coarse_beam=8, leaf_beam=50, top_k=50, batch_size=16),
    )
    atomic_json(output / "demo.json", values)
    config = load_config("Demo", output / "demo.json")
    audit = prepare_events(
        output / "events.csv", output / "features.npz", 480, 600, config.data_directory, "Demo", max_history=8
    )
    result = dict(dataset="Demo", seed=seed, data=audit, config=str(output / "demo.json"))
    if run:
        result["pipeline"] = pipeline(config, "cpu", split="test")
    atomic_json(output / "demo_result.json", result)
    return result
