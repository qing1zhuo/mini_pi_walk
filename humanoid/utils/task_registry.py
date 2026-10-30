import os
from typing import Tuple
from datetime import datetime
from humanoid.algo import VecEnv,OnPolicyRunner

from humanoid import LEGGED_GYM_ENVS_DIR,LEGGED_GYM_ROOT_DIR
from .helpers import *
from humanoid.envs.base.legged_robot_config import *

class TaskRegistry:
    def __init__(self):
        self.task_classes={}
        self.env_cfgs={}
        self.train_cfgs={}

    def register(
        self,
        name:str,
        task_classes:VecEnv,
        env_cfg:LeggedRobotCfg,
        train_cfg:LeggedRobotCfgPPO
    ):
        self.task_classes[name]=task_classes
        self.env_cfgs[name]=env_cfg
        self.train_cfgs[name]=train_cfg

    def get_task_class(self,name:str)->VecEnv:
        return self.task_classes[name]

    def get_cfgs(self,name)->Tuple[LeggedRobotCfg,LeggedRobotCfgPPO]:
        train_cfg=self.train_cfgs[name]
        env_cfg=self.env_cfgs[name]
        env_cfg.seed=train_cfg.seed
        return env_cfg,train_cfg

    def make_env(self,name,args=None,env_cfg=None)->Tuple[VecEnv,LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        if args is None:
            args=get_args()

        if name in self.task_classes:
            task_class=self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")

        if env_cfg is None:
            env_cfg,_=self.get_cfgs(name)
        
        env_cfg,_=update_cfg_from_args(env_cfg,None,args)
        set_seed(env_cfg.seed)

        sim_params={"sim":class_to_dict(env_cfg.sim)}
        sim_params=parse_sim_params(args,sim_params)
        env=task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless
        )

        self.env_cfg_for_wandb=env_cfg
        return env,env_cfg

    def make_alg_runner(self,env,name=None,args=None,train_cfg=None,log_root="default")->Tuple[OnPolicyRunner,LeggedRobotCfg]:
        if args is None:
            args=get_args()

        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _,train_cfg=self.get_cfgs(name)
        else:
            if name is None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        
        _,train_cfg=update_cfg_from_args(None,train_cfg,args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)
        all_cfg = {**train_cfg_dict, **env_cfg_dict}
        
        runner_class = eval(train_cfg_dict["runner_class_name"])
        runner = runner_class(env, all_cfg, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_optimizer=True)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()