"""
Hierarchical encoder for the slim Pokémon Red agent.

The network follows the spec in DESIGN_SLIM_MODEL_V2.md:
CNN -> GRU (short-term) -> LSTM (mid-term) -> SSM (long-term) -> Q-head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HierarchicalState:
    """Container for recurrent state."""

    gru: torch.Tensor
    lstm_h: torch.Tensor
    lstm_c: torch.Tensor
    ssm: torch.Tensor

    def to(self, device: torch.device) -> "HierarchicalState":
        return HierarchicalState(
            gru=self.gru.to(device),
            lstm_h=self.lstm_h.to(device),
            lstm_c=self.lstm_c.to(device),
            ssm=self.ssm.to(device),
        )

    def detach(self) -> "HierarchicalState":
        return HierarchicalState(
            gru=self.gru.detach(),
            lstm_h=self.lstm_h.detach(),
            lstm_c=self.lstm_c.detach(),
            ssm=self.ssm.detach(),
        )

    def mask(self, done: torch.Tensor) -> "HierarchicalState":
        """Zero out finished environments given a done mask shaped (B,) or (B, 1)."""
        if done is None:
            return self
        mask = (1.0 - done.float()).view(done.shape[0], 1)
        return HierarchicalState(
            gru=self.gru * mask,
            lstm_h=self.lstm_h * mask,
            lstm_c=self.lstm_c * mask,
            ssm=self.ssm * mask,
        )

    @classmethod
    def zeros(
        cls, batch_size: int, gru_size: int, lstm_size: int, ssm_size: int, device: torch.device
    ) -> "HierarchicalState":
        return cls(
            gru=torch.zeros(batch_size, gru_size, device=device),
            lstm_h=torch.zeros(batch_size, lstm_size, device=device),
            lstm_c=torch.zeros(batch_size, lstm_size, device=device),
            ssm=torch.zeros(batch_size, ssm_size, device=device),
        )


class CNNEncoder(nn.Module):
    """Lightweight CNN for downsampled Game Boy frames."""

    def __init__(
        self,
        in_channels: int,
        cnn_channels: Tuple[int, ...] = (32, 64, 64),
        kernel_sizes: Tuple[int, ...] = (8, 4, 3),
        strides: Tuple[int, ...] = (4, 2, 1),
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes) == len(strides), "CNN config mismatch"
        layers: list[nn.Module] = []
        prev_c = in_channels
        for out_c, k, s in zip(cnn_channels, kernel_sizes, strides):
            layers.append(nn.Conv2d(prev_c, out_c, kernel_size=k, stride=s))
            layers.append(activation)
            prev_c = out_c
        self.encoder = nn.Sequential(*layers)
        self.output_dim: Optional[int] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.output_dim is None:
            self.output_dim = x.shape[1] * x.shape[2] * x.shape[3]
        return x.flatten(start_dim=1)


class SimpleSSM(nn.Module):
    """
    A lightweight state-space style module.

    Implements a diagonal linear recurrence:
        s_{t+1} = sigmoid(decay) * s_t + (1 - sigmoid(decay)) * W x_t
    """

    def __init__(self, input_size: int, state_size: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.in_proj = nn.Linear(input_size, state_size)
        self.decay = nn.Parameter(torch.zeros(state_size))
        self.norm = nn.LayerNorm(state_size)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, done: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if done is not None:
            mask = (1.0 - done.float()).view(-1, 1)
            state = state * mask
        candidate = self.in_proj(x)
        decay = torch.sigmoid(self.decay).unsqueeze(0)
        next_state = decay * state + (1.0 - decay) * candidate
        return self.norm(next_state), next_state


class SlimHierarchicalQNetwork(nn.Module):
    """
    Hierarchical encoder with CNN -> GRU -> LSTM -> SSM -> Q-head.

    forward(frames, structured, state, done) -> (q_values, new_state, aux)
    """

    def __init__(
        self,
        frame_channels: int,
        num_actions: int,
        structured_dim: int = 0,
        gru_size: int = 128,
        lstm_size: int = 128,
        ssm_size: int = 128,
        cnn_channels: Tuple[int, ...] = (32, 64, 64),
        structured_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(in_channels=frame_channels, cnn_channels=cnn_channels)
        self.structured_dim = structured_dim
        self.structured_encoder = (
            nn.Sequential(
                nn.Linear(structured_dim, structured_hidden),
                nn.LayerNorm(structured_hidden),
                nn.ReLU(inplace=True),
            )
            if structured_dim > 0
            else None
        )

        self.gru_size = gru_size
        self.lstm_size = lstm_size
        self.ssm_size = ssm_size
        self.gru: Optional[nn.GRUCell] = None
        self.lstm: Optional[nn.LSTMCell] = None
        self.ssm = SimpleSSM(input_size=gru_size + lstm_size, state_size=ssm_size)
        self.dropout = nn.Dropout(dropout)

        # defer linear shapes until first forward
        self.q_head = nn.Linear(gru_size + lstm_size + ssm_size, num_actions)

        # set at runtime once CNN output dimension is known
        self._concat_size: Optional[int] = None

    def _prepare_recurrent_cells(self, encoder_out: torch.Tensor) -> None:
        if self._concat_size is not None and self.gru is not None and self.lstm is not None:
            return
        embed_dim = encoder_out.shape[1]
        if self.structured_encoder is not None:
            embed_dim += self.structured_encoder[-3].out_features  # type: ignore[index]
        self._concat_size = embed_dim
        self.gru = nn.GRUCell(input_size=embed_dim, hidden_size=self.gru_size).to(encoder_out.device)
        self.lstm = nn.LSTMCell(input_size=embed_dim, hidden_size=self.lstm_size).to(
            encoder_out.device
        )

    @torch.no_grad()
    def initial_state(self, batch_size: int, device: torch.device) -> HierarchicalState:
        return HierarchicalState.zeros(
            batch_size=batch_size,
            gru_size=self.gru_size,
            lstm_size=self.lstm_size,
            ssm_size=self.ssm_size,
            device=device,
        )

    def forward(
        self,
        frames: torch.Tensor,
        structured: Optional[torch.Tensor],
        state: Optional[HierarchicalState],
        done: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, HierarchicalState, Dict[str, torch.Tensor]]:
        device = frames.device
        batch_size = frames.shape[0]
        frames = self._prepare_frames(frames)
        cnn_out = self.cnn(frames)
        self._prepare_recurrent_cells(cnn_out)

        features = [cnn_out]
        if self.structured_encoder is not None and structured is not None:
            structured = structured.to(device)
            features.append(self.structured_encoder(structured.float()))
        concat = torch.cat(features, dim=-1)
        concat = self.dropout(concat)

        if state is None:
            state = self.initial_state(batch_size, device)

        masked_state = state.mask(done) if done is not None else state
        if self.gru is None or self.lstm is None:
            raise RuntimeError("Recurrent cells were not initialized")
        gru_h = self.gru(concat, masked_state.gru)
        lstm_h, lstm_c = self.lstm(concat, (masked_state.lstm_h, masked_state.lstm_c))
        ssm_out, ssm_state = self.ssm(torch.cat([gru_h, lstm_h], dim=-1), masked_state.ssm, done)

        head_input = torch.cat([gru_h, lstm_h, ssm_out], dim=-1)
        q_values = self.q_head(head_input)

        new_state = HierarchicalState(gru=gru_h, lstm_h=lstm_h, lstm_c=lstm_c, ssm=ssm_state)
        aux = {
            "encoder": cnn_out,
            "gru": gru_h,
            "lstm": lstm_h,
            "ssm": ssm_out,
            "features": concat,
        }
        return q_values, new_state, aux

    @staticmethod
    def _prepare_frames(frames: torch.Tensor) -> torch.Tensor:
        """
        Ensure frames are channel-first float tensors scaled to [0, 1].
        Accepts (B, T, H, W) or (B, H, W, T).
        """
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D frames, got shape {frames.shape}")

        # If an extra leading dim is present (e.g., (B,1,T,H,W)), squeeze it
        if frames.dim() == 5 and frames.shape[1] == 1:
            frames = frames.squeeze(1)

        if frames.shape[1] not in (1, 3, 4, 8):
            # assume channel last
            frames = frames.permute(0, 3, 1, 2)
        return frames.float() / 255.0
