import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from sample_factory.aux_models.aux_model import AuxModel, register_aux_model
from sample_factory.algo.utils.torch_utils import masked_select
from sample_factory.model.encoder import MultiInputEncoder

# This file includes code copied and modified from:
#   https://github.com/ToruOwO/mimex/tree/main/mimex-dmc
#   Copyright (c) Facebook, Inc. and its affiliates.
# Released under the MIT License.
class RNDModel(nn.Module):
    def __init__(self, cfg, obs_space):
        super().__init__()
        self.feature_dim = cfg.rnd_feature_dim
        self.hidden_dim = cfg.rnd_hidden_dim

        self.target_encoder = MultiInputEncoder(cfg, obs_space)
        self.target = nn.Sequential(
            nn.Linear(self.target_encoder.get_out_size(), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )
        self.predictor_encoder = MultiInputEncoder(cfg, obs_space)
        self.predictor = nn.Sequential(
            nn.Linear(self.predictor_encoder.get_out_size(), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )

        # parameter initialization following original paper
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # freeze target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        target_feat = self.target(self.target_encoder(obs))
        pred_feat = self.predictor(self.predictor_encoder(obs))
        return pred_feat, target_feat


class RND(AuxModel):
    def __init__(self, cfg, env_info, learner):
        super().__init__(cfg, env_info, learner)
        self.lr = cfg.rnd_lr
        self.k_expl = cfg.rnd_k_expl
        # Store latest intrinsic reward statistics for summary
        self.last_intrinsic_rewards = None
        self.last_loss = None

    def init(self, device):
        self.rnd = RNDModel(self.cfg, self.env_info.obs_space)
        self.rnd.to(device)
        self.criterion = nn.MSELoss(reduction='none')
        # predictor is optimized
        self.opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=self.lr)
        
    def get_checkpoint_dict(self):
        return {
            "am_rnd_model": self.rnd.state_dict(),
            "am_rnd_optimizer": self.opt.state_dict()
        }

    def load_state(self, checkpoint_dict):
        self.rnd.load_state_dict(checkpoint_dict["am_rnd_model"])
        self.opt.load_state_dict(checkpoint_dict["am_rnd_optimizer"])

    def compute_reward(self, buff):
        # buff.normalized_obs is a dict of tensors.
        # The shape of each tensor is (envs, time, ...).
        # We need to flatten the tensors in buff["obs"] into a single batch dimension,
        # since RND does not exploit the temporal order of the observations.
        obs = dict()
        for key, x in buff["normalized_obs"].items():
            # The rewards are computed relying on next-step observations.
            # obs contains T+1 observations, so we can just take x[:, 1:].
            obs[key] = x[:, 1:].reshape(-1, *x.shape[2:])
        pred_feat, target_feat = self.rnd(obs)

        # update model
        # Don't forget to mask the invalid experiences.
        loss = self.criterion(pred_feat, target_feat).mean(dim=-1)  # (time * envs,)
        valids_flat = buff["valids"][:, :-1].reshape(-1)  # (time * envs,)
        num_invalids = valids_flat.numel() - valids_flat.sum().item()
        masked_loss = masked_select(loss, valids_flat, num_invalids)
        self.opt.zero_grad(set_to_none=True)
        (masked_loss.mean()).backward()
        self.opt.step()

        # Min-Max normalization
        expl_r = loss.detach()
        expl_r = (expl_r - expl_r.min()) / (expl_r.max() - expl_r.min() + 1e-11)
        # The shape of returned intrinsic rewards need to be the same as buff["rewards"]
        expl_r = expl_r.view(*buff["rewards"].shape)
        rewards = expl_r * self.k_expl
        
        # Store rewards and loss for summary
        self.last_intrinsic_rewards = rewards.detach().cpu()
        self.last_loss = masked_loss.mean().detach().cpu()
        
        return rewards, None

    def record_summaries(self):
        if self.last_intrinsic_rewards is None:
            return {}
        
        # Calculate statistics for intrinsic rewards
        rewards_flat = self.last_intrinsic_rewards.flatten()
        return {
            "rnd_reward/mean": float(rewards_flat.mean().item()),
            "rnd_reward/std": float(rewards_flat.std().item()),
            "rnd_reward/min": float(rewards_flat.min().item()),
            "rnd_reward/max": float(rewards_flat.max().item()),
            "rnd_loss": float(self.last_loss.item()),
        }


def make_rnd_reward_generator(cfg, env_info, learner):
    return RND(cfg, env_info, learner)


def register_rnd(method_name="rnd"):
    register_aux_model(method_name, make_rnd_reward_generator)


def add_rnd_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser

    p.add_argument(
        "--rnd_lr",
        default=1e-3,
        type=float,
        help='Learning rate for RND. Default is 1e-3.',
    )
    p.add_argument(
        "--rnd_k_expl",
        default=0.01,
        type=float,
        help="Intrinsic reward scaling coefficient for RND. Default is 0.01.",
    )
    p.add_argument(
        "--rnd_feature_dim",
        default=288,
        type=int,
        help="Feature dimension for RND. Default is 288.",
    )
    p.add_argument(
        "--rnd_hidden_dim",
        default=256,
        type=int,
        help="Hidden dimension for RND. Default is 256.",
    )
