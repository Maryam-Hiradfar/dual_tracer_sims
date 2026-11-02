"""Temporal convolutional network used for dual-tracer TAC separation.

The model consumes tensors shaped ``(batch, input_channels, time)`` and returns
one predicted time series per output channel.  Convolutions are causal, so a
prediction at a given frame cannot use information from a later frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Chomp1d(nn.Module):
    """Remove right padding added by a causal convolution."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        if self.size == 0:
            return x
        return x[..., : -self.size]


class TemporalBlock(nn.Module):
    """Two causal dilated convolutions with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        def convolution(conv_in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(
                    conv_in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.conv1 = convolution(in_channels)
        self.conv2 = convolution(out_channels)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.residual(x))


class Model(nn.Module):
    """A small temporal convolutional network for sequence-to-sequence output.

    Args:
        in_channels: Number of features at each input frame.
        hidden_channels: Width of each temporal block.  A sequence may be used
            to give each block a different width.
        out_channels: Number of output time series (two for dual tracers).
        kernel_size: Width of each temporal convolution.
        num_levels: Number of residual blocks when ``hidden_channels`` is an int.
        dropout: Dropout probability within temporal blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int | Sequence[int] = 16,
        out_channels: int = 2,
        kernel_size: int = 3,
        num_levels: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if isinstance(hidden_channels, int):
            if hidden_channels <= 0 or num_levels <= 0:
                raise ValueError("hidden_channels and num_levels must be positive")
            channels = [hidden_channels] * num_levels
        else:
            channels = list(hidden_channels)
            if not channels or any(channel <= 0 for channel in channels):
                raise ValueError("hidden_channels must contain positive values")

        blocks: list[nn.Module] = []
        block_input = in_channels
        for level, block_output in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    block_input,
                    block_output,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            block_input = block_output

        self.network = nn.Sequential(*blocks)
        self.output_projection = nn.Conv1d(block_input, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "expected input shaped (batch, channels, time), "
                f"but received {tuple(x.shape)}"
            )
        return self.output_projection(self.network(x))


TemporalConvNet = Model


if __name__ == "__main__":
    model = Model()
    sample = torch.randn(4, 1, 100)
    print(model(sample).shape)
