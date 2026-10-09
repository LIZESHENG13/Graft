"""Dataset settings independent of existing checkpoints or installation paths."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

from .io import project_path

ALIASES = {"Video_Games": "Games", "Office_Products": "Office", "Cell_Phones_and_Accessories": "Phones"}
DOMAINS = {value: key for key, value in ALIASES.items()}


@dataclass(frozen=True)
class SearchConfig:
    temperature: float = 0.07
    coarse_weight: float = 0.25
    new_leaf_bias: float = 1.0
    coarse_beam: int = 200
    leaf_beam: int = 50
    top_k: int = 50
    batch_size: int = 64

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if min(self.coarse_beam, self.leaf_beam, self.top_k, self.batch_size) < 1:
            raise ValueError("search budgets must be positive")


def encoder_defaults():
    return dict(
        max_seq_length=20,
        n_layers=2,
        n_heads=2,
        hidden_size=300,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        loss_type="CE",
        train_stage="inductive_ft",
        temperature=0.07,
        adaptor_dropout_prob=0.2,
        adaptor_layers=[768, 300],
        n_exps=8,
    )


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str = "Games"
    domain: str = "Video_Games"
    workspace: Path = field(default_factory=Path.cwd)
    seed: int = 2024
    leaf_budget: int = 2048
    codebook_sizes: tuple[int, int, int] = (144, 144, 144)
    data: str = "data/{domain}"
    cache: str = "artifacts/{dataset}/s{seed}/cache"
    base_catalog: str = "artifacts/{dataset}/s{seed}/base/train_catalog.npz"
    catalog: str = "artifacts/{dataset}/s{seed}/catalog_valid/catalog.npz"
    test_catalog: str = "artifacts/{dataset}/s{seed}/catalog_test/catalog.npz"
    scorer_checkpoint: str = "artifacts/{dataset}/s{seed}/scorer/step_1000.pt"
    encoder_checkpoint: str = "artifacts/{dataset}/s{seed}/encoder/best.pt"
    episodes: str = "artifacts/{dataset}/s{seed}/episodes"
    warm_start: str = "artifacts/{dataset}/s{seed}/warmup/last.pt"
    warmup_steps: int = 500
    scorer_selection_catalog: str | None = None
    hashes: dict[str, str] = field(default_factory=dict)
    encoder: dict = field(default_factory=encoder_defaults)
    encoder_training: dict = field(
        default_factory=lambda: dict(
            epochs=300,
            patience=40,
            lr=0.001,
            batch_size=2048,
            eval_batch_size=1024,
            eval_every=2,
            validation_size=2048,
            selection_metric="ndcg_50",
            bf16=True,
        )
    )
    refinement: dict = field(
        default_factory=lambda: dict(
            steps=1000, base_path_weight=0.0, teacher_weight=100.0, episode_new_only=True, reachable_only=True
        )
    )
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self):
        if len(self.codebook_sizes) != 3 or min(self.codebook_sizes) < 1:
            raise ValueError("codebook_sizes must contain three positive sizes")
        if self.leaf_budget < 0 or self.warmup_steps < 0:
            raise ValueError("leaf_budget and warmup_steps must be nonnegative")

    def path(self, name):
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"No path configured for {name}")
        return project_path(value.format(dataset=self.dataset, domain=self.domain, seed=self.seed), self.workspace)

    @property
    def data_directory(self):
        return self.path("data")


def load_config(dataset="Games", config_path=None, workspace=None, seed=None):
    dataset = ALIASES.get(dataset, dataset)
    if dataset not in DOMAINS and config_path is None:
        raise ValueError("Custom datasets require --config")
    values = {"dataset": dataset, "domain": DOMAINS.get(dataset, dataset)}
    if dataset == "Office":
        values.update(leaf_budget=8192, codebook_sizes=(256, 256, 256), warmup_steps=0)
    elif dataset == "Phones":
        values.update(leaf_budget=8192, codebook_sizes=(288, 288, 288), warmup_steps=0)
    values["search"] = SearchConfig(new_leaf_bias={"Games": 3.0, "Office": 2.0, "Phones": 1.5}.get(dataset, 1.0))
    if config_path:
        path = Path(config_path).expanduser().resolve()
        supplied = json.loads(path.read_text())
        declared = ALIASES.get(supplied.get("dataset", dataset), supplied.get("dataset", dataset))
        if declared != dataset:
            raise ValueError(f"Config dataset {declared} differs from --dataset {dataset}")
        if "workspace" in supplied:
            supplied["workspace"] = project_path(supplied["workspace"], path.parent)
        if "search" in supplied:
            supplied["search"] = replace(values["search"], **supplied["search"])
        if "encoder" in supplied:
            supplied["encoder"] = encoder_defaults() | supplied["encoder"]
        values.update(supplied)
    if workspace is not None:
        values["workspace"] = Path(workspace).expanduser().resolve()
    if seed is not None:
        values["seed"] = seed
    return DatasetConfig(**values)
