import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from sample_factory.model.encoder import Encoder, MlpEncoder
from sample_factory.utils.typing import Config


def _action_dim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Discrete):
        return 1
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    raise NotImplementedError(f"Action space type {type(space)} not supported!")


class _SingleActionEncoder(nn.Module):
    def __init__(self, cfg: Config, act_space: spaces.Space):
        super().__init__()
        self.act_space = act_space
        self.is_discrete = isinstance(act_space, spaces.Discrete)
        self.is_box = isinstance(act_space, spaces.Box)

        if self.is_discrete:
            embed_dim = act_space.n
            self.embed = nn.Embedding(act_space.n, embed_dim)
            mlp_space = spaces.Box(low=-1.0, high=1.0, shape=(embed_dim,), dtype=np.float32)
            self.flat_dim = embed_dim
        elif self.is_box:
            self.flat_dim = _action_dim(act_space)
            mlp_space = spaces.Box(low=-1.0, high=1.0, shape=(self.flat_dim,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Action space type {type(act_space)} not supported!")

        self.mlp = MlpEncoder(cfg, mlp_space)
        self.encoder_out_size = self.mlp.get_out_size()

    def _encode_discrete(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() > 0 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.to(torch.int64)
        return self.embed(actions)

    def _encode_box(self, actions: torch.Tensor) -> torch.Tensor:
        x = actions.float()
        if x.shape[-1] != self.flat_dim:
            if x.dim() < len(self.act_space.shape):
                raise ValueError(f"Unexpected action shape {x.shape} for {self.act_space}")
            leading = x.shape[:-len(self.act_space.shape)]
            x = x.reshape(*leading, self.flat_dim)
        return x

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            x = self._encode_discrete(actions)
        else:
            x = self._encode_box(actions)

        lead_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.mlp(x)
        return x.reshape(*lead_shape, -1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class ActionEncoder(Encoder):
    def __init__(self, cfg: Config, act_space: spaces.Space):
        super().__init__(cfg)
        self.act_space = act_space
        self.is_tuple = isinstance(act_space, spaces.Tuple)

        if self.is_tuple:
            self.encoders = nn.ModuleList([_SingleActionEncoder(cfg, subspace) for subspace in act_space.spaces])
            self.action_splits = [_action_dim(subspace) for subspace in act_space.spaces]
            self.encoder_out_size = sum(enc.get_out_size() for enc in self.encoders)
        else:
            self.encoder = _SingleActionEncoder(cfg, act_space)
            self.encoder_out_size = self.encoder.get_out_size()

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.is_tuple:
            return self.encoder(actions)

        if actions.ndim == 0:
            raise ValueError("Tuple action encoder expects a tensor with at least 1 dimension")

        parts = torch.split(actions, self.action_splits, dim=-1)
        if len(parts) != len(self.encoders):
            raise ValueError(f"Expected {len(self.encoders)} action parts, got {len(parts)}")

        encodings = [enc(part) for enc, part in zip(self.encoders, parts)]
        if len(encodings) == 1:
            return encodings[0]
        return torch.cat(encodings, dim=-1)

    def get_out_size(self) -> int:
        return self.encoder_out_size
