from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .pai.pai_config import PaiCfg, PaiCfgPPO
from .pai.pai_env import PaiFreeEnv

from humanoid.utils.task_registry import task_registry
task_registry.register( "pai_ppo", PaiFreeEnv, PaiCfg(), PaiCfgPPO() )