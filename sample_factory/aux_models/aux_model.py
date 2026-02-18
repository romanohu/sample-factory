from sample_factory.algo.utils.context import global_aux_model_registry
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_dict import TensorDict
from typing import Any, Callable, Dict, List, Optional, Tuple
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import AttrDict, Config
from sample_factory.utils.utils import log
from torch import Tensor

def register_aux_model(method_name: str, make_func: Callable[[Config, EnvInfo], Any]) -> None:
    """
    Register a callable that creates a custom aux model.
    """
    registry = global_aux_model_registry()

    if method_name in registry:
        log.warning(f"Aux model {method_name} already registered, overwriting...")

    assert callable(make_func), f"{make_func=} must be callable"

    registry[method_name] = make_func


def create_aux_models(method_name_raw: Optional[str], cfg: Config, env_info: EnvInfo, learner) -> Any:
    """
    Factory function that creates (multiple) aux model instances.
    """

    if method_name_raw is None:
        return []

    registry = global_aux_model_registry()

    # method_name can be a comma-separated list of aux models
    aux_models = []
    method_names = method_name_raw.split(',')
    for method_name in method_names:
        if method_name not in registry:
            msg = f"Aux model {method_name} not registered. See register_aux_model()!"
            log.error(msg)
            log.debug(f"Registered aux model names: {registry.keys()}")
            raise ValueError(msg)

        make_func = registry[method_name]
        aux_model = make_func(cfg, env_info, learner)
        aux_models.append(aux_model)

    return aux_models


class AuxModel(Configurable):
    def __init__(self, cfg: Config, env_info: EnvInfo, learner):
        super().__init__(cfg)
        self.env_info = env_info
        self.learner = learner

    def init(self, device):
        """
        Initialize the aux model.
        """
        pass

    def extra_buffer_requirements(self) -> List[str]:
        """
        Return a list of extra buffer fields required by the aux model.
        Supported fields are:
        - "normalized_obs_tp1": next-step normalized observations
        - "actions_tp1": next-step actions
        """
        return []
        
    def get_checkpoint_dict(self):
        """
        Get checkpoint dictionary for the aux model.
        """
        return {}

    def load_state(self, checkpoint_dict):
        """
        Load state from checkpoint dictionary.
        """
        pass

    def compute_reward(self, buff: TensorDict) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Compute intrinsic rewards for the given buffer.
        This function is called in Learner._calculate_rewards().
        Tensors in `buff` has the shape of (n_envs, n_steps, ...),
        which is different from `buff` in `compute_loss()`.

        Returns: A tuple of (aux_rewards, valid_rewards)
        aux_rewards (Optional[Tensor]): The computed intrinsic rewards with shape (n_envs, n_steps).
        valid_rewards (Optional[Tensor]): A boolean tensor with shape (n_envs, n_steps) indicating valid experiences.
        If certain experiences are invalid for computing intrinsic rewards,
        the corresponding entries in `valid_rewards` should be set to False.
        """
        return None, None

    def compute_loss(self, buff: AttrDict, actor_critic,
                     head_outputs: Tensor, core_outputs: Tensor,
                     action_logits: Tensor, values: Tensor
                     ) -> Optional[Tensor]:
        """
        Compute aux loss for the given buffer.
        This function is called in Learner._calculate_losses().
        `buff` does not preserve the order of the observations and actions.
        (Tensors in `buff` have the shape of (time * n_envs, ...).)
        If the aux model needs next-step observations or actions, include
        `"normalized_obs_tp1"` and `"actions_tp1"` in `extra_buffer_requirements()`.

        Returns: The computed aux loss tensor.
        """
        return None
    
    def record_summaries(self) -> Dict[str, float]:
        """
        Record summaries for the aux model.
        """
        return {}
