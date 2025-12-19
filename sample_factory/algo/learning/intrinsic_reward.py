from sample_factory.algo.utils.context import global_intrinsic_reward_generator_registry
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_dict import TensorDict
from typing import Any, Callable, Dict, Optional
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log
from torch import Tensor

def register_intrinsic_reward_generator(method_name: str, make_func: Callable[[Config, EnvInfo], Any]) -> None:
    """
    Register a callable that creates a custom intrinsic reward generator.
    """
    registry = global_intrinsic_reward_generator_registry()

    if method_name in registry:
        log.warning(f"Intrinsic reward generator {method_name} already registered, overwriting...")

    assert callable(make_func), f"{make_func=} must be callable"

    registry[method_name] = make_func


def create_intrinsic_reward_generator(method_name: Optional[str], cfg: Config, env_info: EnvInfo, learner) -> Any:
    """
    Factory function that creates intrinsic reward generator instances.
    """

    if method_name is None:
        return IntrinsicRewardGenerator(cfg, env_info, learner)

    registry = global_intrinsic_reward_generator_registry()

    if method_name not in registry:
        msg = f"Intrinsic reward generator {method_name} not registered. See register_intrinsic_reward_generator()!"
        log.error(msg)
        log.debug(f"Registered intrinsic reward generator names: {registry.keys()}")
        raise ValueError(msg)

    make_func = registry[method_name]
    intrinsic_reward_generator = make_func(cfg, env_info, learner)

    return intrinsic_reward_generator


class IntrinsicRewardGenerator(Configurable):
    def __init__(self, cfg: Config, env_info: EnvInfo, learner):
        super().__init__(cfg)
        self.env_info = env_info
        self.learner = learner

    def init(self, device):
        """
        Initialize the intrinsic reward generator.
        """
        pass
        
    def get_checkpoint_dict(self):
        """
        Get checkpoint dictionary for the intrinsic reward generator.
        """
        return {}

    def load_state(self, checkpoint_dict):
        """
        Load state from checkpoint dictionary.
        """
        pass

    def generate_reward(self, buff: TensorDict) -> Optional[Tensor]:
        """
        Generate intrinsic reward for the given buffer.
        """
        return None

    def record_summaries(self) -> Dict[str, float]:
        """
        Record summaries for the intrinsic reward generator.
        """
        return {}
