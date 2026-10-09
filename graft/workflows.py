"""Reproducible registration, staged training, and component controls."""

from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from .catalog import Catalog
from .data import catalog_metadata
from .io import atomic_json, fresh_directory, sha256
from .registration import register_arrivals


def register(config, output, device="cpu", split="valid", budget=None):
    if split not in ("valid", "test"):
        raise ValueError("Registration split must be valid or test")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    base = Catalog.load(config.path("base_catalog"))
    metadata = catalog_metadata(config)
    cutoff = json.loads((config.data_directory / "audit.json").read_text())["timestamp_cutoffs_ms"][split]
    vectors = np.load(config.path("cache") / "item_vectors.npy", allow_pickle=False)
    limit = config.leaf_budget if budget is None else budget
    grown = register_arrivals(base, vectors, cutoff, metadata["stable_order"], limit, device)
    # Both snapshots replay from the initial catalog, so the budget is cumulative.
    if split == "test" and budget is None and config.path("catalog").exists():
        earlier = Catalog.load(config.path("catalog"))
        prior = earlier.codes[:, 0] >= 0
        np.testing.assert_array_equal(grown.codes[prior], earlier.codes[prior])
    output = fresh_directory(output)
    grown.save(output / "catalog.npz")
    result = dict(
        status="completed",
        dataset=config.dataset,
        seed=config.seed,
        split=split,
        budget=limit,
        cutoff=int(cutoff),
        arrivals=len(grown.append_trace),
        new_leaves=sum(x["fresh_path_admitted"] for x in grown.append_trace),
        existing_addresses_unchanged=True,
        base_catalog_sha256=sha256(config.path("base_catalog")),
    )
    atomic_json(output / "result.json", result)
    return result


def pipeline(config, device="cpu", until="evaluate", split="test", resume=False):
    from .encoder import cache_features
    from .evaluation import run_evaluation
    from .quantization import build_base_catalog
    from .refinement import RefinementConfig, train_refinement
    from .train_encoder import train_encoder
    from .training import TrainingConfig, train_scorer
    from .training_episodes import prepare_episodes

    stages = ["encoder", "cache", "quantize", "register", "episodes", "warmup", "train", "evaluate"]
    if until not in stages:
        raise ValueError(f"Unknown pipeline stage {until}")
    rows = []

    def execute(stage, folder, action):
        done = Path(folder) / "result.json"
        if resume and done.exists():
            record = json.loads(done.read_text())
            if record.get("status") != "completed" or record.get("seed", config.seed) != config.seed:
                raise ValueError(f"Cannot resume incompatible artifact: {done}")
            rows.append({"stage": stage, "status": "reused"})
        else:
            rows.append({"stage": stage, "result": action()})
        return stage == until

    if execute(
        "encoder",
        config.path("encoder_checkpoint").parent,
        lambda: train_encoder(config, config.path("encoder_checkpoint").parent, device),
    ):
        return rows
    if execute("cache", config.path("cache"), lambda: cache_features(config, config.path("cache"), device)):
        return rows
    if execute(
        "quantize",
        config.path("base_catalog").parent,
        lambda: build_base_catalog(
            config, config.path("cache") / "item_vectors.npy", config.path("base_catalog").parent
        ),
    ):
        return rows
    if execute(
        "register",
        config.path("catalog").parent,
        lambda: register(config, config.path("catalog").parent, device, "valid"),
    ):
        return rows
    if execute("episodes", config.path("episodes"), lambda: prepare_episodes(config, config.path("episodes"), device)):
        return rows
    if config.warmup_steps:
        if execute(
            "warmup",
            config.path("warm_start").parent,
            lambda: train_scorer(
                config, config.path("warm_start").parent, device, TrainingConfig(steps=config.warmup_steps)
            ),
        ):
            return rows
    elif until == "warmup":
        return rows
    refinement = RefinementConfig(**config.refinement)
    if execute(
        "train",
        config.path("scorer_checkpoint").parent,
        lambda: train_refinement(
            config,
            config.path("scorer_checkpoint").parent,
            device,
            refinement,
            config.path("episodes"),
            warm_start=config.path("warm_start") if config.warmup_steps else None,
        ),
    ):
        return rows
    if split == "test":
        cache_features(config, config.path("cache"), device, splits=("test",))
        execute(
            "register-test",
            config.path("test_catalog").parent,
            lambda: register(config, config.path("test_catalog").parent, device, "test"),
        )
    evaluation = config.path("scorer_checkpoint").parent.parent / f"evaluation_{split}"
    execute("evaluate", evaluation, lambda: run_evaluation(config, evaluation, device, split))
    return rows


def ablate(config, output, device="cpu", split="test", no_graft_checkpoint=None):
    from .evaluation import run_evaluation

    output = fresh_directory(output)
    reference = replace(config, search=replace(config.search, new_leaf_bias=1.0))
    rows = []
    variants = [
        ("graft_b1", {}),
        ("graft_b0", {"bias": 0}),
        ("identity", {"identity_query": True}),
        ("parent", {"parent_descriptor": True}),
        ("parent_b0", {"parent_descriptor": True, "bias": 0}),
    ]
    reuse = output / "reuse_catalog"
    register(config, reuse, device, split, budget=0)
    variants.append(("reuse_b0", {"catalog_path": reuse / "catalog.npz", "bias": 0}))
    if no_graft_checkpoint:
        variants.append(("no_graft", {"scorer_path": no_graft_checkpoint}))
    for name, kwargs in variants:
        result = run_evaluation(reference, output / name, device, split=split, **kwargs)
        rows.append(dict(variant=name, result_file=f"{name}/result.json", metrics=result["metrics"]))
    result = dict(
        status="completed", dataset=config.dataset, seed=config.seed, split=split, reference_bias=1, variants=rows
    )
    atomic_json(output / "result.json", result)
    return result
