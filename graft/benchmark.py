"""Measure frozen retrieval latency and profiler-reported matrix FLOPs."""

from dataclasses import replace
import time
import numpy as np
import torch

from .evaluation import load_retriever
from .io import atomic_json, fresh_directory


def benchmark(config, output, device="cpu", split="test", queries=256, repeats=3, warmup=20, batch_size=64):
    if min(queries, repeats, batch_size) < 1 or warmup < 0:
        raise ValueError("Invalid benchmark budgets")
    retriever, _ = load_retriever(config, device, split=split)
    values = np.load(config.path("cache") / f"{split}_queries.npy", allow_pickle=False)
    with np.load(config.data_directory / f"p2_{split}.npz", allow_pickle=False) as file:
        timestamps = file["timestamp"]
    chosen = np.sort(np.random.default_rng(2024).choice(len(values), min(queries, len(values)), replace=False))
    values = torch.as_tensor(values[chosen], device=device).float()
    times = torch.as_tensor(timestamps[chosen], device=device)
    search = replace(config.search, batch_size=batch_size)

    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            retriever.recommend(values[:1], times[:1], search)
        latencies = []
        for _ in range(repeats):
            for i in range(len(values)):
                synchronize()
                start = time.perf_counter()
                retriever.recommend(values[i : i + 1], times[i : i + 1], search)
                synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
        throughput = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            for i in range(0, len(values), batch_size):
                retriever.recommend(values[i : i + batch_size], times[i : i + batch_size], search)
            synchronize()
            throughput.append(len(values) / (time.perf_counter() - start))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
            retriever.recommend(values[:1], times[:1], search)
        flops = sum(event.flops for event in profile.key_averages())
    result = dict(
        dataset=config.dataset,
        seed=config.seed,
        split=split,
        device=str(device),
        scope="cached-query path retrieval and item resolution; history/text encoding excluded",
        queries=len(values),
        repeats=repeats,
        batch_size=batch_size,
        latency_mean_ms=float(np.mean(latencies)),
        latency_p95_ms=float(np.quantile(latencies, 0.95)),
        throughput_queries_per_second=float(np.mean(throughput)),
        profiler_matrix_gflops=flops / 1e9,
        flop_scope="PyTorch-profiler supported matrix operations, not an all-operator FLOP count",
    )
    output = fresh_directory(output)
    atomic_json(output / "result.json", result)
    return result
