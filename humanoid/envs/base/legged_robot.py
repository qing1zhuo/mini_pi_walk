# self.root_states：机器人根刚体（base_link）的状态，本项目形状为 [num_envs, 13]。
# 第 0 维为环境索引；第 1 维为以下 13 个状态分量（索引从 0 开始）：
# 索引  分量  含义                                      单位
#   0   x     世界坐标系中的 x 位置                     m
#   1   y     世界坐标系中的 y 位置                     m
#   2   z     世界坐标系中的 z 位置（竖直向上）         m
#   3   qx    姿态四元数的 x 分量                      无量纲
#   4   qy    姿态四元数的 y 分量                      无量纲
#   5   qz    姿态四元数的 z 分量                      无量纲
#   6   qw    姿态四元数的 w 分量                      无量纲
#   7   vx    世界坐标系中的 x 方向线速度              m/s
#   8   vy    世界坐标系中的 y 方向线速度              m/s
#   9   vz    世界坐标系中的 z 方向线速度              m/s
#  10   wx    世界坐标系中的角速度 x 分量              rad/s
#  11   wy    世界坐标系中的角速度 y 分量              rad/s
#  12   wz    世界坐标系中的角速度 z 分量              rad/s
# 分组切片：位置 [:, 0:3]，姿态 [:, 3:7]，线速度 [:, 7:10]，角速度 [:, 10:13]。
# 四元数顺序为 [x, y, z, w]，表示机身相对于世界坐标系的姿态，不是欧拉角。
# base_lin_vel / base_ang_vel 通过 quat_rotate_inverse 转为机身坐标系中的速度。
# z 是世界坐标系高度，不一定等于离地高度；角速度分量一般不等于欧拉角变化率。
# 关节角度和关节速度另存于 dof_pos / dof_vel，不包含在这 13 维中。


import os 
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.base_task import BaseTask

from humanoid.utils.math import *
from humanoid.utils.helpers import *
from .legged_robot_config import LeggedRobotCfg

def get_euler_xyz_tensor(quat):
    r,p,w=get_euler_xyz(quat)
    euler_xyz=torch.stack((r,p,w),dim=1)
    euler_xyz[euler_xyz>np.pi]-=2*np.pi
    return euler_xyz

class LeggedRobot(BaseTask):
    def __init__(
        self,
        cfg:LeggedRobotCfg,         # 机器人环境配置
        sim_params,physics_engine,  # 仿真参数与物理引擎
        sim_device,
        headless
    ):
        self.cfg=cfg
        self.sim_params=sim_params
        self.height_samples=None
        self.debug_viz=False
        self.init_done=False

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg,sim_params,physics_engine,sim_device,headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos,self.cfg.viewer.lookat)

        # 创建仿真以取得关节、刚体和接触力张量
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done=True

    def step(self,actions):
        clip_actions=self.cfg.normalization.clip_actions
        self.actions=torch.clip(
            actions,-clip_actions,clip_actions
        ).to(self.device)

        self.render()

        for _ in range(self.cfg.control.decimation):
            # 根据动作计算关节
            self.torques=self._compute_torques(self.actions).view(self.torques.shape)
            # 计算接触力            
            self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device=="cpu":
                self.gym.fetch_results(self.sim,True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # 刷新，计算奖励，打终止标记，生成下一步观测
        self.post_physics_step()

        clip_obs=self.cfg.normalization.clip_observations
        self.obs_buf=torch.clip(self.obs_buf,-clip_obs,clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf=torch.clip(self.privileged_obs_buf,-clip_obs,clip_obs)
        
        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.reset_buf,self.extras

    def reset(self):
        # 重置所有环境，并送入零动作并走两步，返回初始观测
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        obs,privileged_obs,_,_,_=self.step(torch.zeros(self.num_envs,self.num_actions,device=self.device,requires_grad=False))
        return obs,privileged_obs

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf+=1
        self.common_step_counter+=1

        # 奖世界坐标系的重力速度转换为机器人自身坐标系
        self.base_quat[:]=self.root_states[:,3:7]
        self.base_lin_vel[:]=quat_rotate_inverse(self.base_quat,self.root_states[:,7:10])
        self.base_ang_vel[:]=quat_rotate_inverse(self.base_quat,self.root_states[:,10:13])
        self.projected_gravity[:]=quat_rotate_inverse(self.base_quat,self.gravity_vec)
        self.base_euler_xyz=get_euler_xyz_tensor(self.base_quat)

        # 更新速度指令、地形高度、根节点速度扰动等等
        self._post_physics_step_callback()

        # 先判断是否终止并计算奖励
        self.check_termination()
        self.compute_reward()
        # 获取终止的环境的编号
        env_ids=self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # 计算最新状态的观测
        self.compute_observations()


        # 保存上一时刻的状态
        self.last_last_actions[:]=torch.clone(self.last_actions[:])
        self.last_actions[:]=self.actions[:]
        self.last_dof_vel[:]=self.dof_vel[:] 
        self.last_root_vel[:]=self.root_states[:,7:13]
        self.last_rigid_state[:]=self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    
    def check_termination(self):
        # 对于环境中需要检测的刚体，找出接触力大于1的，则该环境reset
        self.reset_buf=torch.any(
            torch.norm(self.contact_forces[:,self.termination_contact_indices,:],dim=-1)>1.,
            dim=1
        )
        # 计算超时buf, 并加入重置buf
        self.time_out_buf=self.episode_length_buf>self.max_episode_length
        self.reset_buf|=self.time_out_buf

    def reset_idx(self,env_ids):
        if len(env_ids)==0:
            return
        # 若启用课程学习，则按照表现调整地形难度和速度指令范围
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # 若启用指令课程且达一定步数
        if self.cfg.commands.curriculum and (self.common_step_counter% self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # 重置关键和根状态，重新采样指令
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # 清零动作
        self.last_last_actions[env_ids]=0.0     
        self.actions[env_ids]=0.0               # 动作清零
        self.last_actions[env_ids]=0.0          # 上一刻动作清零
        self.last_rigid_state[env_ids]=0.0      # 上一刻刚体状态
        self.last_dof_vel[env_ids]=0.0          # 上一刻关节速度
        self.feet_air_time[env_ids]=0.0         # 脚滞空时间
        self.episode_length_buf[env_ids]=0.0    # 轨迹长度
        self.reset_buf[env_ids]=1.0             # 把对应环境reset标记
        self.random_half_phase[0,env_ids]=torch.pi*torch.randint(0,2,(len(env_ids),),device=self.device)

        # 记录reward
        if getattr(self,"record_eval_rewards",False):
            self.eval_episode_ids=env_ids.clone()
            self.eval_episode_rewards={
                key:values[env_ids].clone() for key,values in self.episode_sums.items()
            }
        # 记录episode
        self.extras['episode']={}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_"+key]=torch.mean(self.episode_sums[key][env_ids])/self.max_episode_length_s
            self.episode_sums[key][env_ids]=0.
        # 记录地形trimesh和指令课程情况
        if self.cfg.terrain.mesh_type=="trimesh":
            self.extras["episode"]["terrain_level"]=torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]['max_command_x']=self.command_ranges["lin_vel_x"][1]
        # 记录是否超时
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"]=self.time_out_buf
        # 环境重置后，重新计算各种速度
        self.base_quat[env_ids]=self.root_states[env_ids,3:7]
        self.base_euler_xyz=get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids]=quat_rotate_inverse(self.base_quat[env_ids],self.gravity_vec[env_ids])

    def compute_reward(self):
        self.rew_buf[:]=0.

        for i in range(len(self.reward_functions)):
            # 取出该项奖励名字，并计算该项奖励
            name=self.reward_names[i]
            rew=self.reward_functions[i]()*self.reward_scales[name]
            self.rew_buf+=rew
            self.episode_sums[name]+=rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:]=torch.clip(self.rew_buf[:],min=0.)

        # 计算终止奖励
        if "termination" in self.reward_scales:
            rew=self._reward_termination()*self.reward_scales["termination"]
            self.rew_buf+=rew
            self.episode_sums["termination"]+=rew

    def set_camera(self,postion,lookat):
        cam_pos=gymapi.Vec3(postion[0],postion[1],postion[2])
        cam_target=gymapi.Vec3(lookat[0],lookat[1],lookat[2])
        self.gym.viewer_camera_look_at(self.viewer,None,cam_pos,cam_target)

    # 初始化刚体属性 —— 摩擦
    def _process_rigid_shape_props(self,props,env_id):
        # 如果有域随机化
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                friction_range=self.cfg.domain_rand.friction_range
                num_buckets=256
                bucket_ids=torch.randint(0,num_buckets,(self.num_envs,1))
                friction_buckets=torch_rand_float(friction_range[0],friction_range[1],(num_buckets,1),device='cpu')
                self.friction_coeffs=friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction=self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self,props,env_id):
        if env_id==0:
            self.dof_pos_limits=torch.zeros(self.num_dof,2,dtype=torch.float,device=self.device,requires_grad=False)
            self.dof_vel_limits=torch.zeros(self.num_dof,dtype=torch.float,device=self.device,requires_grad=False)
            self.torque_limits=torch.zeros(self.num_dof,dtype=torch.float,device=self.device,requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i,0]=props["lower"][i].item()*self.cfg.safety.pos_limit
                self.dof_pos_limits[i,1]=props["upper"][i].item()*self.cfg.safety.pos_limit
                self.dof_vel_limits[i]=props["velocity"][i].item()*self.cfg.safety.vel_limit
                self.torque_limits[i]=props["effort"][i].item()*self.cfg.safety.torque_limit

        return props

    def _process_rigid_body_props(self,props,env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng=self.cfg.domain_rand.added_mass_range
            props[0].mass+=np.random.uniform(rng[0],rng[1])
        return props

    def _post_physics_step_callback(self):
        env_ids=(self.episode_length_buf%int(self.cfg.commands.resampling_time/self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # 朝向模式下，把目标朝向与当前朝向之差转为偏航角速度
        if self.cfg.commands.heading_command:
            forward=quat_apply(self.base_quat,self.forward_vec)
            heading=torch.atan2(forward[:,1],forward[:,0])
            self.commands[:,2]=torch.clip(0.5*wrap_to_pi(self.commands[:,3]-heading),-1.,1,)

        # 崎岖模式下，额外采样周围地面高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights=self._get_heights()

        # 定期改变机器人根节点速度
        if self.cfg.domain_rand.push_robots and (self.common_step_counter%self.cfg.domain_rand.push_interval==0): # TODO PR
            self._push_robots()

    def _resample_commands(self,env_ids):
        self.commands[env_ids,0]=torch_rand_float(self.command_ranges["lin_vel_x"][0],self.command_ranges["lin_vel_x"][1],(len(env_ids),1),device=self.device).squeeze(1)
        self.commands[env_ids,1]=torch_rand_float(self.command_ranges["lin_vel_y"][0],self.command_ranges["lin_vel_y"][1],(len(env_ids),1),device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids,3]=torch_rand_float(self.command_ranges["heading"][0],self.command_ranges["heading"][1],(len(env_ids),1),device=self.device).squeeze(1)
        else:
            self.commands[env_ids,2]=torch_rand_float(self.command_ranges["ang_vel_yaw"][0],self.command_ranges["ang_vel_yaw"][1],(len(env_ids),1),device=self.device).squeeze(1)

        self.commands[env_ids,:2]*=(torch.norm(self.commands[env_ids,:2],dim=1)>self.cfg.commands.lin_vel_dead_limit).unsqueeze(1)

    def _compute_torques(self,actions):
        """将动作解释为关节目标角偏移，经 PD 控制并裁剪后返回力矩。"""
        # 目标关节角 = 默认角 + 动作缩放；PD 力矩再按关节上限裁剪
        actions_scaled=actions*self.cfg.control.action_scale
        p_gains=self.p_gains
        d_gains=self.d_gains
        torques=p_gains*(actions_scaled+self.default_dof_pos-self.dof_pos)-d_gains*self.dof_vel

        return torch.clip(torques,-self.torque_limits,self.torque_limits)

    def _reset_dofs(self,env_ids):
        """将指定环境的关节位置设为默认角 ±0.1 rad，速度清零。"""
        # 关节位置在默认角附近扰动，速度清零，再写回 Isaac Gym
        self.dof_pos[env_ids]=self.default_dof_pos+torch_rand_float(-0.1,0.1,(len(env_ids),self.num_dof),device=self.device)
        self.dof_vel[env_ids]=0.

        env_ids_int32=env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(env_ids_int32),len(env_ids_int32))

    def _reset_root_states(self,env_ids):
        """将指定环境的根节点放回出生点；复杂地形上随机偏移 XY。
        按各环境的地形原点放置机器人；复杂地形上再随机偏移 XY"""
        self.root_states[env_ids]=self.base_init_state
        self.root_states[env_ids,:3]+=self.env_origins[env_ids]
        if self.custom_origins:
            self.root_states[env_ids,:2]+=torch_rand_float(-1.0,1.0,(len(env_ids),2),device=self.device)

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids,7:13]=0.0
            self.root_states[env_ids,2]+=1.8
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device)

        env_ids_int32=env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(env_ids_int32),len(env_ids_int32))

    def _update_terrain_curriculum(self,env_ids):
        if not self.init_done:
            return 
        distance=torch.norm(self.root_states[env_ids,:2]-self.env_origins[env_ids,:2],dim=1)

        move_up=distance>self.terrain.env_length/2
        move_down=(distance<torch.norm(self.commands[env_ids,:2],dim=1)*self.max_episode_length_s*0.5) & ~move_up
        self.terrain_levels[env_ids]+=1*move_up-1*move_down

        # 超过最高等级时随机回到某个等级，同时更新环境出生位置
        self.terrain_levels[env_ids]=torch.where(
            self.terrain_levels[env_ids]>=self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids],self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids],0)
        )
        self.env_origins[env_ids]=self.terrain_origins[self.terrain_levels[env_ids],self.terrain_types[env_ids]]

    def update_command_curriculum(self,env_ids):
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])/self.max_episode_length>0.8*self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0]=np.clip(self.command_ranges["lin_vel_x"][0]-0.5,-self.cfg.commands.max_curriculum,0.)
            self.command_ranges["lin_vel_x"][1]=np.clip(self.command_ranges["lin_vel_x"][1]+0.5, 0., self.cfg.commands.max_curriculum,)
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # 获取 Isaac Gym 管理的根节点、关节、接触力和刚体状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 包装仿真张量，并切出关节位置、速度和接触力等视图
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, 13, 13)

        # 初始化控制、指令、历史动作和地形采样所需的缓冲区
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_pos = self.root_states[:, 0:3]
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # 按关节名称匹配配置中的默认角度、PD 刚度和阻尼
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(name)
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        # Actor 与 Critic 分别保存多帧观测历史，供具体环境拼接输入
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale=self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key]*=self.dt
            
        self.reward_functions=[]
        self.reward_names=[]
        for name,scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name='_reward_'+name
            self.reward_functions.append(getattr(self,name))

        self.episode_sums={
            name:torch.zeros(self.num_envs,dtype=torch.float,device=self.device,requires_grad=False) for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        plane_params=gymapi.PlaneParams()
        plane_params.normal=gymapi.Vec3(0.0,0.0,1.0)
        plane_params.static_friction=self.cfg.terrain.static_friction
        plane_params.dynamic_friction=self.cfg.terrain.dynamic_friction
        plane_params.restitution=self.cfg.terrain.restitution
        self.gym.add_ground(self.sim,plane_params)

    def _create_heightfield(self):
        # 将地形高度栅格及其尺度传给仿真器
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        # 将三角网格顶点和面片传给仿真器
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # 从配置指定的 URDF/MJCF 路径加载机器人模型
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 根据刚体名称确定脚、膝盖及碰撞惩罚/终止所需的刚体
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 为每个并行环境分配出生位置
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        for i in range(self.num_envs):
            # 创建环境实例和机器人，并应用摩擦、关节、质量等属性
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 将刚体名称转换为仿真张量中的索引，供接触判断和奖励使用
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # 复杂地形使用地形生成器给出的出生点和难度等级
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # 平面地形把机器人按网格摆放，避免不同环境重叠
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        # 一个策略步长 = 物理步长 × decimation；据此换算回合和扰动间隔
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # 将采样到的地形高度点画在交互窗口中
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        # 在机器人坐标系中构造用于测量地形高度的二维采样网格
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """按机器人位置和朝向采样周围地形高度；平面地形返回零。"""
        # 平面高度恒为零；无地形时无法测量高度
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 按机器人朝向旋转采样点，再移动到世界坐标系
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # 世界坐标转为高度图索引，并限制在有效范围内
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # 取相邻三个栅格点中的最小高度，再换算为米
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
