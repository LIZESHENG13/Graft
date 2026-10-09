#!/usr/bin/env python3
"""Run independent dataset/seed jobs, with one active job per device."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=("Games", "Office", "Phones"), default=["Games", "Office", "Phones"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--config-directory", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.devices)) != len(args.devices):
        parser.error("Seed and device lists must not contain duplicates")
    workspace = args.workspace.resolve()
    jobs = [(dataset, seed) for dataset in args.datasets for seed in args.seeds]
    queues = [[] for _ in args.devices]
    for index, (dataset, seed) in enumerate(jobs):
        device_index = index % len(args.devices)
        command = [
            sys.executable,
            "-m",
            "graft",
            "pipeline",
            "--dataset",
            dataset,
            "--config",
            str((args.config_directory / f"{dataset.lower()}.json").resolve()),
            "--workspace",
            str(workspace),
            "--seed",
            str(seed),
            "--device",
            args.devices[device_index],
            "--split",
            args.split,
        ]
        if args.resume:
            command.append("--resume")
        queues[device_index].append((dataset, seed, command))
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    logs = workspace / "artifacts" / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run_queue(queue):
        for dataset, seed, command in queue:
            log = logs / f"{dataset}_s{seed}_{args.split}.log"
            with log.open("a" if args.resume else "x") as handle:
                subprocess.run(command, cwd=workspace, stdout=handle, stderr=subprocess.STDOUT, check=True)
            print(f"Completed {dataset}, seed {seed}: {log}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = [pool.submit(run_queue, queue) for queue in queues if queue]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
