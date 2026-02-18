from typing import Any, Dict

from sample_factory.model.model_factory import ModelFactory
from sample_factory.utils.typing import CreateEnvFunc, CreateAgentPolicyMappingFunc


class SampleFactoryContext:
    def __init__(self):
        self.env_registry = dict()
        self.model_factory = ModelFactory()
        self.agent_policy_mapping_registry = dict()
        self.aux_model_registry = dict()


GLOBAL_CONTEXT = None


def sf_global_context() -> SampleFactoryContext:
    global GLOBAL_CONTEXT
    if GLOBAL_CONTEXT is None:
        GLOBAL_CONTEXT = SampleFactoryContext()
    return GLOBAL_CONTEXT


def set_global_context(ctx: SampleFactoryContext):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = ctx


def reset_global_context():
    """
    Most useful in tests, call this after any part of the global context has been modified
    by a test in any way.
    """
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = SampleFactoryContext()


def global_env_registry() -> Dict[str, CreateEnvFunc]:
    """
    :return: global env registry
    :rtype: EnvRegistry
    """
    return sf_global_context().env_registry


def global_model_factory() -> ModelFactory:
    """
    :return: global model factory
    :rtype: ModelFactory
    """
    return sf_global_context().model_factory


def global_agent_policy_mapping_registry() -> Dict[str, CreateAgentPolicyMappingFunc]:
    """
    :return: global agent policy mapping registry
    :rtype: AgentPolicyMappingRegistry
    """
    return sf_global_context().agent_policy_mapping_registry


def global_aux_model_registry() -> Dict[str, Any]:
    """
    :return: global aux model registry
    :rtype: AuxModelRegistry
    """
    return sf_global_context().aux_model_registry
