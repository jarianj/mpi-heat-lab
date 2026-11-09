"""Training loop with validation-based model selection and multiple seeds."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..solver import MODELS as MODEL_DIR
from ..solver import RESULTS
from .dataset import load_split
from .models import MODELS, count_parameters


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(prediction - target, dim=(-2, -1))
    denominator = torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(1e-8)
    return (numerator / denominator).mean()


def to_tensors(split: dict, target_device: torch.device) -> tuple[torch.Tensor, ...]:
    initial = torch.from_numpy(split["initial"]).to(target_device)
    alpha = torch.from_numpy(split["alpha"]).to(target_device)
    change = torch.from_numpy(split["target"]).to(target_device) - initial
    return initial, alpha, change


def train_model(
    name: str, seed: int, epochs: int = 150, batch_size: int = 16, lr: float = 1e-3
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = device()
    train = to_tensors(load_split("train"), target_device)
    val = to_tensors(load_split("val"), target_device)
    model = MODELS[name]().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    count = train[0].shape[0]
    batches = int(np.ceil(count / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * batches
    )

    best = float("inf")
    history = []
    path = MODEL_DIR / f"{name}_seed{seed}.pt"
    MODEL_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(count, device=target_device)
        total = 0.0
        for index in range(batches):
            chosen = permutation[index * batch_size : (index + 1) * batch_size]
            loss = relative_l2(model(train[0][chosen], train[1][chosen]), train[2][chosen])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * chosen.numel()
        model.eval()
        with torch.no_grad():
            predictions = torch.cat(
                [
                    model(val[0][i : i + 32], val[1][i : i + 32])
                    for i in range(0, val[0].shape[0], 32)
                ]
            )
            val_loss = relative_l2(predictions, val[2]).item()
        history.append({"epoch": epoch, "train": total / count, "val": val_loss})
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), path)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"{name} seed={seed} epoch={epoch} train={total / count:.4f} val={val_loss:.4f}",
                flush=True,
            )
    return {
        "model": name,
        "seed": seed,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "best_val": best,
        "train_seconds": time.perf_counter() - start,
        "device": str(target_device),
        "history": history,
    }


def train_all(seeds: int = 3, epochs: int = 150) -> None:
    runs = []
    RESULTS.mkdir(exist_ok=True)
    for name in MODELS:
        for seed in range(seeds):
            runs.append(train_model(name, seed, epochs))
            (RESULTS / "training.json").write_text(json.dumps(runs, indent=2))


def load_model(name: str, seed: int = 0, target_device: torch.device | None = None):
    target_device = target_device or device()
    model = MODELS[name]().to(target_device)
    state = torch.load(
        MODEL_DIR / f"{name}_seed{seed}.pt", map_location=target_device, weights_only=True
    )
    model.load_state_dict(state)
    return model.eval()
