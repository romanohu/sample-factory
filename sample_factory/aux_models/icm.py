import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from sample_factory.aux_models.aux_model import AuxModel, register_aux_model
from sample_factory.aux_models.action_encoder import ActionEncoder
from sample_factory.algo.utils.torch_utils import masked_select
from sample_factory.model.encoder import MultiInputEncoder


class ICMModel(nn.Module):
    def __init__(self, cfg, obs_space, act_space):
        super().__init__()
        self.obs_encoder = MultiInputEncoder(cfg, obs_space)
        self.act_encoder = ActionEncoder(cfg, act_space)

        self.obs_feature_dim = self.obs_encoder.get_out_size()
        self.act_feature_dim = self.act_encoder.get_out_size()
        self.hidden_dim = cfg.icm_hidden_dim
        self.inverse_dm_input_dim = self.obs_feature_dim * 2
        self.forward_dm_input_dim = self.obs_feature_dim + self.act_feature_dim

        self.inverse_dm = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.inverse_dm_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.act_feature_dim)
        )
        self.forward_dm = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.forward_dm_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.obs_feature_dim)
        )

        # parameter initialization
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, obs, next_obs, action):
        obs_feat = self.obs_encoder(obs)
        next_obs_feat = self.obs_encoder(next_obs)
        action_feat = self.act_encoder(action)
        pred_next_obs_feat = self.forward_dm(torch.cat([obs_feat, action_feat], dim=-1))
        pred_action_feat = self.inverse_dm(torch.cat([obs_feat, next_obs_feat], dim=-1))
        return obs_feat, next_obs_feat, action_feat, pred_next_obs_feat, pred_action_feat


class ICM(AuxModel):
    def __init__(self, cfg, env_info, learner):
        super().__init__(cfg, env_info, learner)
        self.lr = cfg.icm_lr
        self.k_expl = cfg.icm_k_expl
        # Store latest intrinsic reward statistics for summary
        self.last_intrinsic_rewards = None
        self.last_inverse_dm_loss = None
        self.last_forward_dm_loss = None

    def init(self, device):
        self.icm = ICMModel(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        self.icm.to(device)
        self.criterion = nn.MSELoss(reduction='none')
        self.opt = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

    def get_checkpoint_dict(self):
        return {
            "am_icm_model": self.icm.state_dict(),
            "am_icm_optimizer": self.opt.state_dict()
        }

    def load_state(self, checkpoint_dict):
        self.icm.load_state_dict(checkpoint_dict["am_icm_model"])
        self.opt.load_state_dict(checkpoint_dict["am_icm_optimizer"])

    def compute_reward(self, buff):
        # buff.normalized_obs is a dict of tensors.
        # The shape of each tensor is (envs, time, ...).
        # We need to flatten the tensors in buff["obs"] into a single batch dimension.
        obs = dict()
        next_obs = dict()
        for key, x in buff["normalized_obs"].items():
            obs[key] = x[:, :-1].reshape(-1, *x.shape[2:])
            # obs contains T+1 observations, so we can just take x[:, 1:].
            next_obs[key] = x[:, 1:].reshape(-1, *x.shape[2:])
        actions = buff["actions"].reshape(-1, *buff["actions"].shape[2:])

        # time where dones is True are not valid
        valids_flat = buff["valids"][:, :-1].reshape(-1)  # (time * envs,)
        valids_flat &= ~buff["dones"].reshape(-1)

        (obs_feat, 
         next_obs_feat, 
         action_feat, 
         pred_next_obs_feat, 
         pred_action_feat) = self.icm(obs, next_obs, actions)

        # Compute losses
        forward_loss = self.criterion(pred_next_obs_feat, next_obs_feat).mean(dim=-1)  # (time * envs,)
        inverse_loss = self.criterion(pred_action_feat, action_feat).mean(dim=-1)  # (time * envs,)
        loss = forward_loss + inverse_loss

        # Don't forget to mask the invalid experiences.
        num_invalids = valids_flat.numel() - valids_flat.sum().item()
        masked_loss = masked_select(loss, valids_flat, num_invalids)
        self.opt.zero_grad(set_to_none=True)
        (masked_loss.mean()).backward()
        self.opt.step()

        # Min-Max normalization
        expl_r = forward_loss.detach()
        expl_r = (expl_r - expl_r.min()) / (expl_r.max() - expl_r.min() + 1e-11)
        # The shape of returned intrinsic rewards need to be the same as buff["rewards"]
        expl_r = expl_r.view(*buff["rewards"].shape)
        rewards = expl_r * self.k_expl
        valids = valids_flat.view(*buff["rewards"].shape)
        
        # Store rewards and losses for summary
        self.last_intrinsic_rewards = rewards.detach().cpu()
        self.last_inverse_dm_loss = inverse_loss.mean().detach().cpu()
        self.last_forward_dm_loss = forward_loss.mean().detach().cpu()
        
        return rewards, valids

    def record_summaries(self):
        if self.last_intrinsic_rewards is None:
            return {}
        
        # Calculate statistics for ICM rewards
        rewards_flat = self.last_intrinsic_rewards.flatten()
        return {
            "icm_reward/mean": float(rewards_flat.mean().item()),
            "icm_reward/std": float(rewards_flat.std().item()),
            "icm_reward/min": float(rewards_flat.min().item()),
            "icm_reward/max": float(rewards_flat.max().item()),
            "icm_loss/inverse_dm_loss": float(self.last_inverse_dm_loss.item()),
            "icm_loss/forward_dm_loss": float(self.last_forward_dm_loss.item()),
        }


def make_icm_reward_generator(cfg, env_info, learner):
    return ICM(cfg, env_info, learner)


def register_icm(method_name="icm"):
    register_aux_model(method_name, make_icm_reward_generator)


def add_icm_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser

    p.add_argument(
        "--icm_lr",
        default=1e-3,
        type=float,
        help='Learning rate for ICM. Default is 1e-3.',
    )
    p.add_argument(
        "--icm_k_expl",
        default=0.01,
        type=float,
        help="Intrinsic reward scaling coefficient for ICM. Default is 0.01.",
    )
    p.add_argument(
        "--icm_hidden_dim",
        default=256,
        type=int,
        help="Hidden dimension for dynamics models in ICM. Default is 256.",
    )
