"""Command-line interface for Graft training, registration and evaluation."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import project_path, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "prepare-events",
        "train-encoder",
        "cache",
        "quantize",
        "register",
        "episodes",
        "warmup",
        "train",
        "evaluate",
        "evaluate-predictions",
        "pipeline",
        "ablate",
        "benchmark",
        "doctor",
    ):
        command = commands.add_parser(name)
        command.add_argument("--dataset", default="Games")
        command.add_argument("--config", type=Path)
        command.add_argument("--workspace", type=Path)
        command.add_argument("--seed", type=int)
        command.add_argument("--device", default="cpu")
        command.add_argument("--output", type=Path)
    for name in ("evaluate", "evaluate-predictions", "pipeline", "register", "ablate", "benchmark"):
        commands.choices[name].add_argument(
            "--split", choices=("valid", "test"), default="valid" if name == "register" else "test"
        )
    prepare = commands.choices["prepare"]
    prepare.add_argument(
        "--source", type=Path, required=True, help="Directory containing DOMAIN.data_maps and DOMAIN.sent_emb"
    )
    prepare.add_argument(
        "--benchmarks", type=Path, required=True, help="Directory containing DOMAIN.train/valid/test.csv"
    )
    events = commands.choices["prepare-events"]
    events.add_argument("--events", type=Path, required=True)
    events.add_argument("--features", type=Path, required=True)
    events.add_argument("--train-cutoff", type=int, required=True)
    events.add_argument("--valid-cutoff", type=int, required=True)
    encoder = commands.choices["train-encoder"]
    encoder.add_argument("--epochs", type=int)
    encoder.add_argument("--batch-size", type=int)
    encoder.add_argument("--resume", action="store_true")
    cache = commands.choices["cache"]
    cache.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid"])
    cache.add_argument("--checkpoint", type=Path)
    cache.add_argument("--batch-size", type=int, default=2048)
    commands.choices["quantize"].add_argument("--vectors", type=Path)
    commands.choices["register"].add_argument("--budget", type=int)
    warmup = commands.choices["warmup"]
    warmup.add_argument("--steps", type=int, default=500)
    train = commands.choices["train"]
    train.add_argument("--steps", type=int)
    train.add_argument("--warm-start", type=Path)
    train.add_argument("--episodes", type=Path)
    train.add_argument(
        "--no-graft", action="store_true", help="Matched control: set episode cross-entropy coefficient to zero"
    )
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--bias", type=float)
    evaluate.add_argument("--catalog", type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--identity-query", action="store_true")
    evaluate.add_argument("--parent-descriptor", action="store_true")
    external = commands.choices["evaluate-predictions"]
    external.add_argument("--predictions", type=Path, required=True)
    external.add_argument("--method", required=True)
    pipeline = commands.choices["pipeline"]
    pipeline.add_argument(
        "--until",
        choices=("encoder", "cache", "quantize", "register", "episodes", "warmup", "train", "evaluate"),
        default="evaluate",
    )
    pipeline.add_argument("--resume", action="store_true")
    commands.choices["ablate"].add_argument("--no-graft-checkpoint", type=Path)
    bench = commands.choices["benchmark"]
    bench.add_argument("--queries", type=int, default=256)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--batch-size", type=int, default=64)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--results", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--ddof", type=int, choices=(0, 1), default=1)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=2024)
    demo.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .demo import create_demo

        print(json.dumps(create_demo(args.output, args.seed, not args.prepare_only), indent=2, default=str))
        return
    if args.command == "aggregate":
        from .aggregation import aggregate

        print(json.dumps(aggregate(args.results, args.output, args.ddof), indent=2))
        return
    config = load_config(args.dataset, args.config, args.workspace, args.seed)
    seed_everything(config.seed)
    path = lambda value: project_path(value, config.workspace) if value is not None else None
    default_outputs = {
        "prepare": config.data_directory,
        "prepare-events": config.data_directory,
        "train-encoder": config.path("encoder_checkpoint").parent,
        "cache": config.path("cache"),
        "quantize": config.path("base_catalog").parent,
        "register": config.path("test_catalog" if getattr(args, "split", "valid") == "test" else "catalog").parent,
        "episodes": config.path("episodes"),
        "warmup": config.path("warm_start").parent,
        "train": config.path("scorer_checkpoint").parent,
    }
    output = path(args.output) or default_outputs.get(args.command)
    if args.command not in ("pipeline", "doctor") and output is None:
        parser.error("--output is required for this command")
    if args.command == "prepare":
        from .preprocessing import prepare_amazon

        result = prepare_amazon(
            path(args.source), path(args.benchmarks), config.domain, output, config.encoder["max_seq_length"]
        )
    elif args.command == "prepare-events":
        from .preprocessing import prepare_events

        result = prepare_events(
            path(args.events),
            path(args.features),
            args.train_cutoff,
            args.valid_cutoff,
            output,
            config.domain,
            config.encoder["max_seq_length"],
        )
    elif args.command == "train-encoder":
        from .train_encoder import EncoderTrainingConfig, train_encoder

        values = config.encoder_training.copy()
        for key in ("epochs", "batch_size"):
            if getattr(args, key) is not None:
                values[key] = getattr(args, key)
        result = train_encoder(config, output, args.device, EncoderTrainingConfig(**values), args.resume)
    elif args.command == "cache":
        from .encoder import cache_features

        result = cache_features(config, output, args.device, args.batch_size, path(args.checkpoint), tuple(args.splits))
    elif args.command == "quantize":
        from .quantization import build_base_catalog

        result = build_base_catalog(config, path(args.vectors) or config.path("cache") / "item_vectors.npy", output)
    elif args.command == "register":
        from .workflows import register

        result = register(config, output, args.device, args.split, args.budget)
    elif args.command == "episodes":
        from .training_episodes import prepare_episodes

        result = prepare_episodes(config, output, args.device)
    elif args.command == "warmup":
        from .training import TrainingConfig, train_scorer

        result = train_scorer(config, output, args.device, TrainingConfig(steps=args.steps))
    elif args.command == "train":
        from .refinement import RefinementConfig, train_refinement

        values = config.refinement.copy()
        if args.steps is not None:
            values["steps"] = args.steps
        if args.no_graft:
            values["episode_path_weight"] = 0
        warm = path(args.warm_start) or (config.path("warm_start") if config.warmup_steps else None)
        result = train_refinement(
            config,
            output,
            args.device,
            RefinementConfig(**values),
            path(args.episodes) or config.path("episodes"),
            warm_start=warm,
        )
    elif args.command == "evaluate":
        from .evaluation import run_evaluation

        result = run_evaluation(
            config,
            output,
            args.device,
            args.split,
            args.limit,
            args.bias,
            path(args.catalog),
            path(args.checkpoint),
            args.identity_query,
            args.parent_descriptor,
        )
    elif args.command == "evaluate-predictions":
        from .external_predictions import evaluate_predictions

        result = evaluate_predictions(config, path(args.predictions), output, args.method, args.split)
    elif args.command == "pipeline":
        from .workflows import pipeline

        result = pipeline(config, args.device, args.until, args.split, args.resume)
    elif args.command == "ablate":
        from .workflows import ablate

        result = ablate(config, output, args.device, args.split, path(args.no_graft_checkpoint))
    elif args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(
            config, output, args.device, args.split, args.queries, args.repeats, args.warmup, args.batch_size
        )
    else:
        import torch

        result = dict(
            dataset=config.dataset,
            workspace=str(config.workspace),
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            paths={
                name: {"path": str(config.path(name)), "exists": config.path(name).exists()}
                for name in (
                    "data",
                    "cache",
                    "base_catalog",
                    "catalog",
                    "test_catalog",
                    "encoder_checkpoint",
                    "scorer_checkpoint",
                )
            },
        )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
