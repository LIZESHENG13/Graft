"""First-stage content sequence training; validation selects the checkpoint."""

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from .backbone.model import UniSRec
from .data import load_domain
from .io import atomic_json, seed_everything, sha256
from .metrics import group_metrics


@dataclass(frozen=True)
class EncoderTrainingConfig:
    epochs: int = 300
    patience: int = 40
    lr: float = 0.001
    batch_size: int = 2048
    eval_batch_size: int = 1024
    eval_every: int = 2
    validation_size: int = 2048
    selection_metric: str = "ndcg_50"
    bf16: bool = True

    def __post_init__(self):
        if (
            min(
                self.epochs, self.patience, self.batch_size, self.eval_batch_size, self.eval_every, self.validation_size
            )
            < 1
            or self.lr <= 0
        ):
            raise ValueError("Encoder training budgets and learning rate must be positive")
        if self.selection_metric not in ("recall_50", "ndcg_50"):
            raise ValueError("Select checkpoints by recall_50 or ndcg_50")


def learned_state(model):
    return {key: value for key, value in model.state_dict().items() if key != "plm_embedding.weight"}


@torch.inference_mode()
def evaluate(model, arrays, indices, content, first, seen, limit, batch_size=1024, save=None):
    model.eval()
    vectors = F.normalize(model.moe_adaptor(content[:limit]), dim=-1)
    first_gpu = torch.as_tensor(first[:limit], device=content.device)
    ranks, outputs = [], []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        history = torch.as_tensor(arrays["right_history"][selected], device=content.device, dtype=torch.long)
        lengths = torch.as_tensor(arrays["length"][selected], device=content.device)
        query = F.normalize(model.forward(history, model.moe_adaptor(content[history]), lengths), dim=-1)
        scores = query @ vectors.T
        times = torch.as_tensor(arrays["timestamp"][selected], device=content.device)
        scores[:, 0] = -torch.inf
        scores.masked_fill_(first_gpu[None] > times[:, None], -torch.inf)
        values, ids = scores.topk(min(50, limit), dim=-1)
        prediction = ids.masked_fill(~values.isfinite(), 0)
        prediction = F.pad(prediction, (0, 50 - prediction.shape[1]))
        targets = torch.as_tensor(arrays["item"][selected], device=content.device)
        hit = prediction == targets[:, None]
        rank = torch.where(hit.any(-1), hit.float().argmax(-1) + 1, 0)
        ranks.append(rank.cpu().numpy())
        if save:
            outputs.append(prediction.cpu().numpy().astype(np.int32))
    ranks = np.concatenate(ranks)
    targets = arrays["item"][indices]
    metrics = group_metrics(ranks, ranks > 0, targets, seen)
    for group in metrics.values():
        group.pop("sid_recall_50", None)
    if save:
        np.savez_compressed(save, query_index=indices, item=targets, rank=ranks, items=np.concatenate(outputs))
    return metrics


def train_encoder(dataset, output, device="cpu", settings=None, resume=False):
    settings = settings or EncoderTrainingConfig(**dataset.encoder_training)
    seed_everything(dataset.seed)
    device, output = torch.device(device), Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {output}; use --resume")
    if resume and not (output / "last.pt").exists():
        raise FileNotFoundError("--resume requires last.pt")
    output.mkdir(parents=True, exist_ok=True)
    train, valid, vectors, first, seen, train_limit, valid_limit = load_domain(dataset)
    content = torch.as_tensor(np.array(vectors, copy=True), device=device)
    model_config = dataset.encoder
    if model_config["adaptor_layers"][0] != content.shape[1]:
        raise ValueError("Encoder adaptor input dimension differs from text feature dimension")
    if model_config["adaptor_layers"][-1] != model_config["hidden_size"]:
        raise ValueError("Adaptor output dimension must match the sequence encoder")
    model = UniSRec(model_config, content[:train_limit]).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=settings.lr,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    indices = np.sort(
        np.random.default_rng(2024).choice(
            len(valid["item"]), min(settings.validation_size, len(valid["item"])), replace=False
        )
    )
    np.save(output / "selection_indices.npy", indices)
    atomic_json(
        output / "config.json",
        dict(
            dataset=dataset.dataset,
            seed=dataset.seed,
            encoder=model_config,
            training=settings.__dict__,
            selection_split="valid",
        ),
    )
    batches = {
        key: torch.as_tensor(train[key], device=device, dtype=torch.long) for key in ("right_history", "length", "item")
    }
    start_epoch, best_epoch, best_score = 0, 0, -float("inf")
    if resume:
        state = torch.load(output / "last.pt", map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing != ["plm_embedding.weight"] or unexpected:
            raise ValueError("Resume checkpoint does not match the backbone")
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_epoch, best_score = state["epoch"], state["best_epoch"], state["best_score"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_state"]])
        np.random.set_state(state["numpy_rng_state"])

    def save_checkpoint(path, epoch):
        temporary = path.with_suffix(".tmp")
        torch.save(
            dict(
                model=learned_state(model),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                best_epoch=best_epoch,
                best_score=best_score,
                model_config=model_config,
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all(),
                numpy_rng_state=np.random.get_state(),
            ),
            temporary,
        )
        temporary.replace(path)

    started, progress, epoch = time.monotonic(), [], start_epoch
    for epoch in range(start_epoch + 1, settings.epochs + 1):
        model.train()
        order = torch.randperm(len(train["item"]), device=device)
        total = 0.0
        for start in range(0, len(order), settings.batch_size):
            chosen = order[start : start + settings.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=settings.bf16 and device.type == "cuda"
            ):
                loss = model.calculate_loss(
                    batches["right_history"][chosen], batches["length"][chosen], batches["item"][chosen]
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Encoder loss became nonfinite")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        entry = dict(epoch=epoch, loss=total / ((len(order) + settings.batch_size - 1) // settings.batch_size))
        if epoch % settings.eval_every == 0 or epoch == settings.epochs:
            metrics = evaluate(model, valid, indices, content, first, seen, valid_limit, settings.eval_batch_size)
            score = metrics["overall"][settings.selection_metric]
            if score > best_score:
                best_score, best_epoch = score, epoch
                save_checkpoint(output / "best.pt", epoch)
            entry.update(validation=metrics, best_epoch=best_epoch)
        save_checkpoint(output / "last.pt", epoch)
        progress.append(entry)
        atomic_json(output / "progress.json", progress)
        print(entry, flush=True)
        if best_epoch and epoch - best_epoch >= settings.patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    metrics = evaluate(
        model,
        valid,
        np.arange(len(valid["item"])),
        content,
        first,
        seen,
        valid_limit,
        settings.eval_batch_size,
        output / "valid_predictions.npz",
    )
    result = dict(
        status="completed",
        dataset=dataset.dataset,
        seed=dataset.seed,
        trained_epochs=epoch,
        selected_epoch=best_epoch,
        selection_split="valid",
        metrics=metrics,
        checkpoint_sha256=sha256(output / "best.pt"),
        seconds=round(time.monotonic() - started, 2),
    )
    atomic_json(output / "result.json", result)
    return result
