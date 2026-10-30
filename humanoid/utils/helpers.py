import os
import copy
import torch
import numpy as np
import random

from isaacgym import gymapi
from isaacgym import gymutil

from humanoid import LEGGED_GYM_ENVS_DIR,LEGGED_GYM_ROOT_DIR

def class_to_dict(obj)->dict:
    # 如果对象中没有字典，直接返回
    if not hasattr(obj,"__dict__"):
        return obj

    result={}
    # 把对象作为dir遍历key
    for key in dir(obj):
        # 如果key用'_'开头，直接跳过
        if key.startswith('_'):
            continue

        element=[]
        # 获取key对应的value
        val=getattr(obj,key)
        # 如果value是list, 将其中元素放进element
        if isinstance(val,list):
            for item in val:
                element.append(class_to_dict(item))
        # 若value不是，递归处理
        else:
            element=class_to_dict(val)
        
        result[key]=element

    return result

def update_class_from_dict(obj, dict):
    # 遍历字典中的属性名和新值
    for key, val in dict.items():
        # 读取现有属性；不存在则得到 None
        attr = getattr(obj, key, None)

        # 如果现有属性是一个类，就递归更新这个类内部的属性
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            # 否则直接设置属性，通常也可以新增不存在的属性
            setattr(obj, key, val)

    return  # 返回 None，修改已经直接作用于 obj

def set_seed(seed):
    if seed==-1:
        seed=np.random.randint(0,10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args,cfg):
    # args：启动参数对象，提供物理引擎、GPU 开关、子场景数和线程数等。
    # cfg：仿真配置；下面通过 cfg["sim"] 读取其中的仿真参数字典。
    # 创建 Isaac Gym 仿真参数对象，初始值由 SimParams 的默认设置提供。
    sim_params=gymapi.SimParams()

    # 如果选择 Flex 引擎且 device 不是 "cpu"，打印提示；不会自动切换引擎。
    if args.physics_engine==gymapi.SIM_FLEX:
        if args.device!="cpu":
            print("WARNING! Using Flex with GPU instead of PHYSX")
    # 如果选择 PhysX 引擎，设置其专用参数。
    elif args.physics_engine==gymapi.SIM_PHYSX:
        # 本意是设置 PhysX 是否使用 GPU 计算物理。
        # 注意：当前的 == 只比较是否相等，不会赋值；要设置参数应使用 =。
        sim_params.physx.use_gpu=args.use_gpu
        # 设置 PhysX 子场景数量，用于仿真场景的划分。
        sim_params.physx.num_subscenes=args.subscenes
    # 设置是否使用 GPU 数据管线，涉及仿真张量数据所在的设备。
    # 注意：普通字典不支持 cfg.use_gpu_pipeline，需按实际配置来源读取。
    sim_params.use_gpu_pipeline=args.use_gpu_pipeline

    # 若配置包含 sim 部分，用 Isaac Gym 的工具函数将其写入 sim_params。
    # 此处调用的是 gymutil 中的同名函数，不是递归调用当前函数。
    # 配置中出现的同名参数会覆盖前面已经设置的值。
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"],sim_params)

    # 使用 PhysX 且显式指定了正的线程数时，最后覆盖其 CPU 工作线程数。
    # 因为这一步在读取 cfg 之后，所以这里的 num_threads 优先级更高。
    if args.physics_engine==gymapi.SIM_PHYSX and args.num_threads>0:
        sim_params.physx.num_threads=args.num_threads

    return sim_params

def get_load_path(root,load_run=-1,checkpoint=-1):
    try:
        # runs 为root下的文件列表
        runs=os.listdir(root)
        runs.sort()
        # 移除exported，仅保留pt
        if "exported" in runs:
            runs.remove("exported")
        last_run=os.path.join(root,runs[-1])
    except:
        raise ValueError("No runs in this directory: "+root)

    if load_run==-1:
        load_run=last_run
    else:
        load_run=os.path.join(root,load_run)

    if checkpoint==-1:
        models=[file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m:"{0:0>15}".format(m))
        model=models[-1]
    else:
        model="model_{}.pt".format(checkpoint)

    load_path=os.path.join(load_run,model)
    return load_path

def update_cfg_from_args(env_cfg,train_cfg,args):
    if env_cfg is not None:
        if args.num_envs is not None:
            env_cfg.env.num_envs=args.num_envs
    
    if train_cfg is not None:
        if args.seed is not None:
            train_cfg.seed=args.seed
        if args.max_iterations is not None:
            train_cfg.runner.max_iterations=args.max_iterations
        if args.resume:
            train_cfg.runner.resume=args.resume
        if args.experiment_name is not None:
            train_cfg.runner.experiment_name=args.experiment_name
        if args.run_name is not None:
            train_cfg.runner.run_name=args.run_name
        if args.load_run is not None:
            train_cfg.runner.load_run=args.load_run
        if args.checkpoint is not None:
            train_cfg.runner.checkpoint=args.checkpoint
    
    return env_cfg,train_cfg

def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "pai_ppo",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
            "default": "v1",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
    ]

    args=gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id=args.compute_device_id
    args.sim_device=args.sim_device_type
    if args.sim_device=="cuda":
        args.sim_device+=f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic,path):
    os.makedirs(path,exist_ok=True)
    path=os.path.join(path,"policy_torch.pt")
    model=copy.deepcopy(actor_critic).to("cpu")
    dummy_input=torch.randn(1,705)
    trace_script_module=torch.jit.trace(model,dummy_input)
    trace_script_module.save(path)

def export_policy_to_onnx(actor_critic,path):
    os.makedirs(path,exist_ok=True)
    model_path=os.path.join(path,"policy_onnx.onnx")
    model=copy.deepcopy(actor_critic).to("cpu")
    model.eval()

    batch_size=1
    num_observations=705
    dummy_input=torch.randn(batch_size,num_observations)

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"模型已导出为 ONNX 格式，保存路径：{model_path}")