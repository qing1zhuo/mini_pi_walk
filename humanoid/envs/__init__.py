from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .pai.pai_config import PaiCfg, PaiCfgPPO
from .pai.pai_env import PaiFreeEnv
from .gru_pai.pai_config import PaiCfg as GRUPaiCfg
from .gru_pai.pai_config import PaiCfgPPO as GRUPaiCfgPPO
from .gru_pai.pai_env import PaiFreeEnv as GRUPaiFreeEnv

from humanoid.utils.task_registry import task_registry
task_registry.register( "pai_ppo", PaiFreeEnv, PaiCfg(), PaiCfgPPO() )
task_registry.register(
    "gru_pai_ppo", GRUPaiFreeEnv, GRUPaiCfg(), GRUPaiCfgPPO()
)
