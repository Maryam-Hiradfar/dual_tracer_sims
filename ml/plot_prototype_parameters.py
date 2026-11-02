"""Create a compact visual table of the prototype dataset and TCN parameters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


CHECKPOINT = Path("ml/experiments/mixture_30ep/best_tcn_prototype.pt")
DATASET_METADATA = Path("datasets/dual_tracer/dataset_metadata.json")
OUTPUT = Path("ml/experiments/mixture_30ep/prototype_parameters.png")


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = json.loads(DATASET_METADATA.read_text())
    model = checkpoint["model_config"]
    training = checkpoint["training_config"]
    dataset = metadata["dataset_config"]

    rows = [
        ("Dataset", "Total samples", f"{metadata['n_samples']:,}"),
        ("Dataset", "Train / validation / test", "800 / 100 / 100"),
        ("Dataset", "Tracer order", "PBR28 first, FDG second"),
        ("Dataset", "FDG injection delay", ", ".join(f"{x:g}" for x in dataset["delta_choices_min"]) + " min"),
        ("Dataset", "Tracer AIF scale range", f"{dataset['scale_range'][0]:.2f}–{dataset['scale_range'][1]:.2f}"),
        ("Dataset", "Frames / scan duration", f"{len(metadata['frame_midpoints_min'])} / {metadata['frame_edges_min'][-1]:g} min"),
        ("Dataset", "Noise", metadata["noise_model"]),
        ("Dataset", "Dataset / split seeds", f"{metadata['dataset_seed']} / {metadata['split_seed']}"),
        ("TCN", "Input", "1-channel noisy physical mixture"),
        ("TCN", "Output", "2 physical TACs: FDG, PBR28"),
        ("TCN", "Causal convolutions", str(training["causal"])),
        ("TCN", "Hidden channels / levels", f"{model['hidden_channels']} / {model['num_levels']}"),
        ("TCN", "Kernel size / dilations", f"{model['kernel_size']} / 1, 2, 4, 8"),
        ("TCN", "Activation / dropout", f"ReLU / {model['dropout']:g}"),
        ("Training", "Epochs run / best checkpoint", f"30 / epoch {checkpoint['epoch']}"),
        ("Training", "Batch size", str(training["batch_size"])),
        ("Training", "Optimizer / learning rate", f"Adam / {training['learning_rate']:g}"),
        ("Training", "Loss", f"TAC MSE + {training['mixture_weight']:g} × mixture MSE"),
        ("Training", "Normalization scale", f"{checkpoint['normalization_scale']:.5f} kBq/mL"),
        ("Training", "Training seed", str(training["seed"])),
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=("Section", "Parameter", "Prototype value"),
        colWidths=(0.15, 0.38, 0.47),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#315f8f")
            cell.set_text_props(color="white", weight="bold")
        elif rows[row - 1][0] == "TCN":
            cell.set_facecolor("#eef5fb")
        elif rows[row - 1][0] == "Training":
            cell.set_facecolor("#f3f7f1")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "white")
    ax.set_title("Dual-Tracer TCN Prototype Parameters", fontsize=18, weight="bold", pad=18)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
