import random

import numpy as np

from sample_factory.algo.utils.context import global_agent_policy_mapping_registry
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import total_num_envs
from typing import Any, Callable, Dict, List, Optional
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log


def register_agent_policy_mapping(mapping_name: str, make_mapping_func: Callable[[Config, EnvInfo], Any]) -> None:
    """
    Register a callable that creates a custom agent policy mapping.
    """
    mapping_registry = global_agent_policy_mapping_registry()

    if mapping_name in mapping_registry:
        log.warning(f"Agent policy mapping {mapping_name} already registered, overwriting...")

    assert callable(make_mapping_func), f"{make_mapping_func=} must be callable"

    mapping_registry[mapping_name] = make_mapping_func


def create_agent_policy_mapping(mapping_name: Optional[str], cfg: Config, env_info: EnvInfo) -> Any:
    """
    Factory function that creates agent policy mapping instances.
    """

    if mapping_name is None:
        return DefaultAgentPolicyMapping(cfg, env_info)

    mapping_registry = global_agent_policy_mapping_registry()

    if mapping_name not in mapping_registry:
        msg = f"Agent policy mapping {mapping_name} not registered. See register_agent_policy_mapping()!"
        log.error(msg)
        log.debug(f"Registered agent policy mapping names: {mapping_registry.keys()}")
        raise ValueError(msg)

    make_mapping_func = mapping_registry[mapping_name]
    mapping = make_mapping_func(cfg, env_info)

    return mapping


class BaseAgentPolicyMapping:
    def __init__(self, cfg: Config, env_info: EnvInfo):
        pass

    def set_training_info(self, training_info: Dict[PolicyID, Dict[str, Any]]) -> None:
        pass

    def generate_policy_mapping_stats(self, episodic_stats: List[Dict[str, Any]], env_idx: int):
        # This method is expected to return data in the form {POLICY_MAPPING_STATS: data} to be forwarded to Runner.
        # Custom implementations are expected to import POLICY_MAPPING_STATS via
        # `from sample_factory.algo.utils.misc import POLICY_MAPPING_STATS`.
        return None

    def get_policy_for_agent(self, agent_idx: int, env_idx: int, global_env_idx: int) -> int:
        pass


class DefaultAgentPolicyMapping(BaseAgentPolicyMapping):
    """
    This class implements the most simple mapping between agents in the envs and their associated policies.
    We just pick a random policy from the population for every agent at the beginning of the episode.

    Methods of this class can potentially be overloaded to provide a more clever mapping, e.g. we can minimize the
    number of different policies per rollout worker thus minimizing the amount of communication required.
    """

    def __init__(self, cfg: Config, env_info: EnvInfo):
        self.rng = np.random.RandomState(seed=random.randint(0, 2**32 - 1))

        self.num_agents = env_info.num_agents
        self.num_policies = cfg.num_policies
        self.mix_policies_in_one_env = (
            cfg.pbt_mix_policies_in_one_env if hasattr(cfg, "pbt_mix_policies_in_one_env") else False
        )  # TODO

        self.resample_env_policy_every = 10  # episodes
        self.env_policies = dict()
        self.env_policy_requests = dict()

        total_envs = total_num_envs(cfg)
        self.sync_mode = not cfg.async_rl
        if self.sync_mode:
            assert total_envs % self.num_policies == 0, f"{total_envs=} must be divisible by {self.num_policies=}"

    def get_policy_for_agent(self, agent_idx: int, env_idx: int, global_env_idx: int) -> int:
        if self.sync_mode:
            # env_id here is a global index of the policy
            # deterministic mapping ensures we always collect the same amount of experience per policy per iteration
            # Sync mode is an experimental feature. This code can be further improved to allow more sophisticated
            # agent-policy mapping.
            return global_env_idx % self.num_policies

        num_requests = self.env_policy_requests.get(env_idx, 0)

        # extra cheeky flag to make sure this code executes early in the training so we spot any potential problems
        early_in_the_training = num_requests < 5
        if num_requests % (self.num_agents * self.resample_env_policy_every) == 0 or early_in_the_training:
            if self.mix_policies_in_one_env:
                self.env_policies[env_idx] = [self._sample_policy() for _ in range(self.num_agents)]
            else:
                policy = self._sample_policy()
                self.env_policies[env_idx] = [policy] * self.num_agents

        self.env_policy_requests[env_idx] = num_requests + 1
        return self.env_policies[env_idx][agent_idx]

    def _sample_policy(self):
        return self.rng.randint(0, self.num_policies)
