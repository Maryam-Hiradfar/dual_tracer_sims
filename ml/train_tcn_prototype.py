"""Single-file prototype for training the causal dual-tracer TCN.

Run from the repository root:

    conda activate dual_tracer
    python -m ml.train_tcn_prototype --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from ml.tcn import Model


class DualTracerDataset(Dataset):
    """In-memory NPZ dataset returning mixture, tracer targets, and clean sum."""

    def __init__(self, path: Path, scale: float) -> None:
        with np.load(path) as data:
            self.x = torch.from_numpy(
                data["X_measured_tavg_n1_d1"].astype(np.float32)
            )
            self.y = torch.from_numpy(
                data["Y_phys_tavg_n0_d1"].astype(np.float32)
            )
            self.x_clean = torch.from_numpy(
                data["X_clean_tavg_n0_d1"].astype(np.float32)
            )

        if self.x.ndim != 3 or self.x.shape[1] != 1:
            raise ValueError(f"expected X shaped (N, 1, T), received {self.x.shape}")
        if self.y.shape != (len(self.x), 2, self.x.shape[-1]):
            raise ValueError(
                f"expected Y shaped (N, 2, T), received {self.y.shape}"
            )
        if self.x_clean.shape != self.x.shape:
            raise ValueError(
                "X_clean_tavg_n0_d1 must have the same shape as the measured input"
            )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("normalization scale must be finite and positive")

        self.scale = float(scale)
        self.x /= self.scale
        self.y /= self.scale
        self.x_clean /= self.scale

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.x[index], self.y[index], self.x_clean[index]


def training_scale(train_path: Path) -> float:
    """Return one fixed scale derived only from noisy training inputs."""
    with np.load(train_path) as data:
        x_train = data["X_measured_tavg_n1_d1"]
        scale = float(np.max(np.abs(x_train)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("training inputs did not produce a valid normalization scale")
    return scale


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "mps":
            try:
                torch.zeros(1, device=device)
            except RuntimeError as error:
                raise RuntimeError(
                    "PyTorch reports MPS support, but an MPS tensor could not "
                    "be created. Use --device cpu for this environment."
                ) from error
        return device
    if torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            torch.zeros(1, device=device)
            return device
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def batch_loss(
    prediction: Tensor,
    target: Tensor,
    clean_mixture: Tensor,
    *,
    mixture_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    tracer_loss = nn.functional.mse_loss(prediction, target)
    predicted_mixture = prediction.sum(dim=1, keepdim=True)
    mixture_loss = nn.functional.mse_loss(predicted_mixture, clean_mixture)
    total_loss = tracer_loss + mixture_weight * mixture_loss
    return total_loss, tracer_loss, mixture_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    mixture_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "tracer_mse": 0.0, "mixture_mse": 0.0}
    sample_count = 0

    for x, y, x_clean in loader:
        x = x.to(device)
        y = y.to(device)
        x_clean = x_clean.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model(x)
            loss, tracer_loss, mixture_loss = batch_loss(
                prediction,
                y,
                x_clean,
                mixture_weight=mixture_weight,
            )
            if training:
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        sample_count += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["tracer_mse"] += float(tracer_loss.detach().cpu()) * batch_size
        totals["mixture_mse"] += float(mixture_loss.detach().cpu()) * batch_size

    return {name: value / sample_count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    scale: float,
    model_config: dict,
    training_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "normalization_scale": scale,
            "model_config": model_config,
            "training_config": training_config,
            "target": "Y_phys_tavg_n0_d1",
            "channel_order": ["FDG", "PBR28"],
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/dual_tracer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/prototype_outputs"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mixture-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Train and validate on the same first N training samples.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test split; leave off during model development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0 or args.mixture_weight < 0:
        raise ValueError("learning rate must be positive and mixture weight non-negative")

    set_seed(args.seed)
    device = choose_device(args.device)
    train_path = args.data_dir / "train.npz"
    scale = training_scale(train_path)

    train_dataset = DualTracerDataset(train_path, scale)
    val_dataset = DualTracerDataset(args.data_dir / "val.npz", scale)
    test_dataset = DualTracerDataset(args.data_dir / "test.npz", scale)

    if args.overfit_samples:
        count = min(args.overfit_samples, len(train_dataset))
        indices = list(range(count))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = Subset(train_dataset.dataset, indices)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model_config = {
        "in_channels": 1,
        "hidden_channels": args.hidden_channels,
        "out_channels": 2,
        "kernel_size": args.kernel_size,
        "num_levels": args.num_levels,
        "dropout": args.dropout,
    }
    model = Model(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "mixture_weight": args.mixture_weight,
        "seed": args.seed,
        "causal": True,
    }

    checkpoint_path = args.output_dir / "best_tcn_prototype.pt"
    history = []
    best_val_loss = float("inf")

    print(f"device={device}")
    print(f"normalization_scale={scale:.6g}")
    print(
        f"train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            mixture_weight=args.mixture_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            mixture_weight=args.mixture_weight,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss']:.6g} "
            f"val={val_metrics['loss']:.6g} "
            f"val_mix={val_metrics['mixture_mse']:.6g}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val_loss,
                scale=scale,
                model_config=model_config,
                training_config=training_config,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
        file.write("\n")

    if args.evaluate_test:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loader = make_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            optimizer=None,
            mixture_weight=args.mixture_weight,
        )
        with (args.output_dir / "test_metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(test_metrics, file, indent=2)
            file.write("\n")
        print(f"test={test_metrics['loss']:.6g}")

    print(f"best_checkpoint={checkpoint_path}")
    print(f"best_val_loss={best_val_loss:.6g}")


if __name__ == "__main__":
    main()
