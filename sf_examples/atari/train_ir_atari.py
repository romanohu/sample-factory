import sys

from sample_factory.aux_models.rnd import register_rnd, add_rnd_env_args
from sample_factory.aux_models.icm import register_icm, add_icm_env_args
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.atari.atari_params import atari_override_defaults
from sf_examples.atari.atari_utils import ATARI_ENVS, make_atari_env


def register_atari_envs():
    for env in ATARI_ENVS:
        register_env(env.name, make_atari_env)


def register_atari_components():
    register_atari_envs()


def register_intrinsic_reward_generators():
    register_rnd(method_name="rnd")
    register_icm(method_name="icm")


def parse_atari_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    atari_override_defaults(partial_cfg.env, parser)
    add_rnd_env_args(parser)
    add_icm_env_args(parser)
    parser.set_defaults(
        aux_models="rnd,icm",
    )
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_atari_components()
    register_intrinsic_reward_generators()
    cfg = parse_atari_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
