import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

class BaseTask():
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        sim_device,
        headless
    ):
        self.gym=gymapi.acquire_gym()   # 获取gym对象，用于后续环境创建、机器人加载、仿真执行

        self.sim_params=sim_params  # 仿真参数
        self.physics_engine=physics_engine  # 物理引擎
        self.sim_device=sim_device  # 仿真设备
        sim_device_type,self.sim_device_id=gymutil.parse_device_str(self.sim_device)    # 仿真设备与gpu id
        self.headless=headless # 是否无头

        # 总体设备
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device=self.sim_device
        else:
            self.device="cpu"

        self.graphics_device_id=self.sim_device_id  # 图形渲染设备

        self.num_envs=cfg.env.num_envs  # 环境数量
        self.num_obs=cfg.env.num_observations
        self.num_privileged_obs=cfg.env.num_privileged_obs
        self.num_actions=cfg.env.num_actions

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # 分配缓冲区
        self.obs_buf=torch.zeros(
            self.num_envs,self.num_obs,device=self.device,dtype=torch.float
        )
        self.rew_buf=torch.zeros(
            self.num_envs,device=self.device,dtype=torch.float
        )
        self.neg_reward_buf=torch.zeros(
            self.num_envs,device=self.device,dtype=torch.float
        )
        self.pos_reward_buf=torch.zeros(
            self.num_envs,device=self.device,dtype=torch.float
        )
        self.reset_buf=torch.ones(
            self.num_envs,device=self.device,dtype=torch.long
        )
        self.episode_length_buf=torch.zeros(
            self.num_envs,device=self.device,dtype=torch.long
        )
        self.time_out_buf=torch.zeros(
            self.num_envs,device=self.device,dtype=torch.bool
        )
        # 起始随机步态
        self.random_half_phase=torch.pi*torch.randint(
            0,2,(1,self.num_envs),device=self.device,dtype=torch.long
        )

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf=torch.zeros(
                self.num_envs,self.num_privileged_obs,device=self.device,dtype=torch.float
            )
        else:
            self.privileged_obs_buf=None

        self.extras={}

        self.create_sim()               # 创建仿真实例，并保存为self.sim
        self.gym.prepare_sim(self.sim)  # 让isaacgym初始化tensor api
        self.enable_viewer_sync=True    # 查看窗口同步开关
        self.viewer=None                # 窗口属性


        if self.headless==False:
            # 为仿真sim创建可视化窗口
            self.viewer=self.gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            # ESC则“QUIT”; V则“toggle_viewer_sync”
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer,gymapi.KEY_ESCAPE,"QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer,gymapi.KEY_V,"toggle_viewer_sync"
            )
        
        # TODO PR
        # 创建相机传感器，设置相机输出图像的分辨率
        camera_properties=gymapi.CameraProperties()
        camera_properties.width=720
        camera_properties.height=480
        # 在第一个仿真环境中船舰相机，并保存其句柄
        camera_handle=self.gym.create_camera_sensor(
            self.envs[0],camera_properties
        )
        self.camera_handle=camera_handle

    def get_observations(self):
        return self.obs_buf
    def get_privileged_observations(self):
        return self.privileged_obs_buf
    def get_rma_observations(self):
        return self.rma_obs_buf

    def reset_idx(self,env_ids):
        raise NotImplementedError
    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        obs,privileged_obs,_,_,_ =self.step(
            torch.zeros(
                self.num_envs,self.num_actions,device=self.device,requires_grad=False
            )
        )
        return obs,privileged_obs

    def step(self,actions):
        raise NotImplementedError

    def render(self,sync_frame_time=True):
        """处理窗口事件并刷新画面；sync_frame_time 控制是否等待以匹配真实时间。

        本函数不推进物理仿真，也不渲染或读取相机传感器图像。
        """
        # 只有创建了可视化窗口才执行；无头模式下 viewer 为 None，直接跳过。
        if self.viewer:
            # 用户关闭窗口时退出程序，而不只是结束本次 render 调用。
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            
            # 读取初始化时注册的按键事件；value > 0 表示按下，避免松开时再次触发。
            for evt in self.gym.query_viewer_action_events(self.viewer):
                # Esc：退出程序。
                if evt.action=="QUIT" and evt.value>0:
                    sys.exit()
                # V：切换是否刷新窗口画面；关闭刷新并不等于暂停物理仿真。
                elif evt.action=="toggle_viewer_sync" and evt.value>0:
                    self.enable_viewer_sync=not self.enable_viewer_sync
            
            # 非 CPU 路径在绘制前等待物理仿真结果就绪；True 表示阻塞等待。
            if self.device!="cpu":
                self.gym.fetch_results(self.sim,True)
            
            if self.enable_viewer_sync:
                # 将最新物理状态（如物体位置和姿态）同步到图形系统，不推进物理仿真。
                self.gym.step_graphics(self.sim)
                # 将当前场景绘制到可视化窗口。
                self.gym.draw_viewer(self.viewer,self.sim,True)
                # 仿真快于真实时间时等待，使显示节奏接近实时；慢于实时则不会加速。
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                # 不刷新画面时仍处理窗口事件，保持响应，以便后续再次按 V 恢复刷新。
                self.gym.poll_viewer_events(self.viewer)
