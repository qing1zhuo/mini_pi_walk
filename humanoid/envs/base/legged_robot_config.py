from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs=4096
        num_observations=235
        num_privileged_obs=None
        num_actions=12
        env_spacing=3.0
        send_timeouts=True
        episode_length_s=20     # 每个episode长度20s

    class terrain:
        # 地形碰撞表示：'plane' 为平面，'heightfield' 为高度场，'trimesh' 为三角网格。
        # 参考 PaiEnv.create_sim() 用 Python None 表示不创建地面，不接受字符串 'none'。
        mesh_type='trimesh'
        # 高度图在 x、y 方向的栅格间距，单位 m；0.1 表示每格 10 cm。
        # 越小能表示越细的地形，但固定面积下栅格数和网格开销越大。
        horizontal_scale=0.1
        # 高度图整数值对应的高度单位，单位 m；实际高度 = 原始整数 × 此值。
        # 0.005 表示高度量化精度为 5 mm；障碍物实际高度由生成函数另行决定。
        vertical_scale=0.005
        # 地形块阵列四周额外添加的零高度边界宽度，单位 m；不是机器人之间的间距。
        border_size=25
        # 启用地形课程：按行生成递增难度，重置时根据机器人离出生点的水平距离升降级。
        # 距离超过 terrain_length / 2 升级；未升级且距离小于
        # 指令平面速度大小 × 回合时长 × 0.5 时降级。平面模式会关闭此开关。
        curriculum=True
        # 地面静摩擦系数（无量纲），影响接触面开始相对滑动的难易程度。
        static_friction=1.0
        # 地面动摩擦系数（无量纲），影响接触面已经滑动时的摩擦阻力。
        # 这两项设置地面材质；机器人碰撞体的摩擦还可由 domain_rand 单独随机化。
        dynamic_friction=1.0
        # 接触恢复系数（无量纲，通常 0~1）；0 表示非弹性碰撞，越大反弹越明显。
        restitution=0.0

        # 是否采样机器人周围地面的高度；参考实现从高度图查询，不是模拟深度相机。
        # 是否送入 Actor/Critic 由环境的观测拼接决定；更改采样点时须检查观测维度。
        # 平面模式采样结果为 0；没有地面时应关闭高度采样。
        measure_heights=True
        # 相对机身的前后方向采样坐标，单位 m；正 x 向前，范围为前后各 0.8 m。
        measured_points_x=[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
        # 相对机身的左右方向采样坐标，单位 m；正 y 向左，范围为左右各 0.5 m。
        # x、y 做笛卡尔积，共 17 × 11 = 187 点，覆盖 1.6 m × 1.0 m，包含中心线。
        # 采样区域跟随机身平移和偏航角旋转，不跟随横滚角、俯仰角倾斜。
        measured_points_y=[-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 是否指定单一地形生成函数；参考实现优先判断 curriculum，因此要使用
        # selected 必须先关闭 curriculum；两者均 False 时随机生成类型与难度。
        selected=False
        # 指定地形的函数名及参数；selected=False 时不使用。
        # 参考 selected_terrain() 从 'type' 取函数名、从嵌套 terrain_kwargs 取参数。
        # 该分支还引用了未初始化的 self.vertical_scale/self.horizontal_scale，启用前需修正。
        terrain_kwargs=None
        # 课程开启时的初始难度等级上限（包含上限）；5 表示从 0~5 随机选择。
        # 必须小于 num_rows；关闭课程后，参考实现改为从所有行随机选择初始位置。
        max_init_terrain_level=5
        # 单块地形沿世界 x 方向的长度，单位 m；不是整张地图的长度。
        terrain_length=8.
        # 单块地形沿世界 y 方向的宽度，单位 m；当前每块为 8 m × 8 m。
        # 参考生成器用 width 同时构造两个方向，长宽设为不同值前需检查生成与拼接逻辑。
        terrain_width=8.
        # num_rows：地形行数，课程模式下对应难度等级，difficulty = 行号 / num_rows。
        # num_cols：地形列数，课程模式下按比例分配类型；列数不等于不同类型的数量。
        # 地形块总数为 num_rows × num_cols，与并行机器人数量 num_envs 不同。
        # 不含边界的地图大小为 (num_rows × terrain_length) × (num_cols × terrain_width)。
        # 各类型的比例，按累计值划分选择区间，应非负且总和为 1。
        # 课程模式按列分配，比例受列数离散化影响；随机模式表示各类型的抽样概率。
        # 若使用参考 Terrain.make_terrain()，当前五项实际对应：
        # 平滑坡 10%、粗糙坡 10%、负台阶高度楼梯 35%、正台阶高度楼梯 25%、离散障碍 20%。
        # 注意参考配置注释的 stairs up/down 顺序与代码中的台阶高度符号相反。
        # 若使用 PaiEnv 的 HumanoidTerrain，顺序则是：平地、离散障碍、随机起伏、
        # 正坡、负坡、正台阶、负台阶；不能直接沿用上面的五类含义。
        num_rows=10
        num_cols=10
        terrain_proportions=[0.1,0.1,0.35,0.25,0.2]

        # 仅在高度图转三角网格时生效：将超过阈值的陡坡修正为竖直面，改善台阶边缘。
        # 阈值为坡度 |高度差 / 水平距离|，不是角度；0.75 对应约 36.9°。
        # slope_treshold 是参考代码实际使用的拼写，不能单独改成 slope_threshold。
        slope_treshold=0.75

    class commands:
        curriculum=False
        max_curriculum=1
        num_commands=4
        resampling_time=10
        heading_command=True    # 是否从朝向误差中计算角速度命令
        lin_vel_dead_limit=0.05

        class ranges:
            lin_vel_x=[-1.0,1.0]
            lin_vel_y=[-1.0,1.0]
            ang_vel_yaw=[-1,1]
            heading=[-3.1415926,3.1415926]

    class init_state:
        pos=[0.0, 0.0, 1.]  # 初始位置(x,y,z)
        rot=[0.0, 0.0, 0.0, 1.0]    # 初始旋转角
        lin_vel=[0.0,0.0,0.0]
        ang_vel=[0.0,0.0,0.0]
        default_joint_angles={
            "joint_a":0.,
            "joint_b":0.
        }

    class control:
        # 参考 LeggedRobot._compute_torques()：策略动作先转换为目标关节角，再用 PD 算力矩。
        # q_target = default_joint_angles + action_scale × action
        # torque = Kp × (q_target - 当前关节角) - Kd × 当前关节角速度
        # 此处目标角速度为 0，最终力矩按机器人各关节的 torque_limits 裁剪。
        # 字典键按关节名称的子串匹配；joint_a/joint_b 是示例，需对应实际机器人关节。
        # stiffness、damping 应使用相同的键；未匹配关节在参考实现中 Kp、Kd 均设为 0。
        # 避免多个键匹配同一关节，否则后匹配的参数会覆盖先匹配的参数。

        # 比例增益 Kp（关节刚度），单位 N·m/rad：位置误差越大，恢复力矩越大。
        # 例如 Kp=10、角度误差=0.1 rad 时，比例项产生 1 N·m 力矩。
        # 增大通常使位置跟踪更强，但过大可能引起振荡或使力矩频繁达到上限。
        stiffness={'joint_a':10.0, 'joint_b':15.0}
        # 微分增益 Kd（关节阻尼），单位 N·m·s/rad：产生与关节转动方向相反的力矩。
        # 用于抑制快速运动和振荡；过大可能使动作迟缓，需要与 Kp 配合调整。
        damping={'joint_a':1.0,'joint_b':1.5}

        # 将无量纲动作缩放为相对默认关节角的偏移，转动关节的偏移单位为 rad。
        # 0.5 表示 action=1 时目标角为默认角 +0.5 rad，action=0 时目标角为默认角。
        # 参考基类先按 normalization.clip_actions 裁剪动作，再做此缩放；动作不保证在 [-1, 1]。
        # 此参数控制目标角偏移幅度，不是力矩上限，也不负责限制关节角范围。
        action_scale=0.5

        # 每个策略步执行的物理仿真步数，必须为正整数；2 表示一次动作持续 2 个物理步。
        # 期间动作/目标角保持不变，但每个物理步都根据最新关节状态重新计算 PD 力矩。
        # 策略周期 = decimation × sim.dt；策略频率 = 1 / (decimation × sim.dt)。
        # 例如 sim.dt=0.005 s、decimation=2 时，物理与 PD 更新频率为 200 Hz，策略频率为 100 Hz。
        # 固定 sim.dt 时，增大会降低策略更新频率；它与物理求解器的 sim.substeps 不同。
        decimation=2

    class asset:
        file=""
        name="legged_robot"
        foot_name="None"

        penalize_contacts_on=[]
        terminate_after_contacts_on=[]
        disable_gravity=False

        collapse_fixed_joints=True
        fix_base_link=False
        default_dof_drive_mode=3
        self_collisions=0
        replace_cylinder_with_capsule=True
        flip_visual_attachments=True

        density=0.001
        angular_damping=0.
        linear_damping=0.
        max_angular_velocity=1000.
        max_linear_velocity=1000.
        armature=0.
        thickness=0.01

    class domain_rand:
        randomize_friction=True
        friction_range=[0.5,1.25]
        randomize_base_mass=False
        added_mass_range=[-1.,1.]
        push_robots=True
        push_interval_s=15
        max_push_vel_xy=1.

    class rewards:
        class scales:
            termination=-0.0
            tracking_lin_vel=1.0
            tracking_ang_vel=0.5
            lin_vel_z=-2.0
            ang_vel_xy=-0.05
            orientation=-0.
            torques=-0.00001
            dof_vel=-0.
            dof_acc=-2.5e-7
            base_height=-0.
            feet_air_time=1.0
            collision=-1
            feet_stumble=-0.0
            action_rate=-0.0
            stand_still=-0.0

        only_positive_rewards=True
        tracking_sigma=0.25
        max_contact_force=100.0

    class normalization:
        class obs_scales:
            lin_vel=2.0
            ang_vel=0.25
            dof_pos=1.0
            dof_vel=0.05
            height_measurements=5.0
        clip_observations=100.0
        clip_actions=100.

    class noise:
        add_noise=True
        noise_level=1.0
        class noise_scales:
            dof_pos=0.01
            dof_vel=1.5
            lin_vel=0.1
            ang_vel=0.2
            gravity=0.05
            height_measurements=0.1

    class viewer:
        ref_env=0
        pos=[10,0,6]
        lookat=[11.,5.,3.]

    class sim:
        # 一次 gym.simulate() 推进的仿真时间，单位 s；0.005 s 对应每秒 200 个仿真步。
        # 这是模拟世界的时间步长，不保证程序按真实时间运行；越小通常越精细但计算量越大。
        # 策略周期 = dt × control.decimation；当前 decimation=2，策略周期为 0.01 s（100 Hz）。
        dt=0.005
        # 每个 dt 内部划分的物理子步数，正整数；子步时长 = dt / substeps。
        # 当前为 1，即每个子步 0.005 s；增大可改善数值稳定性，但会增加计算开销。
        # 不改变策略周期；参考控制循环在每次 simulate() 前算一次 PD，而非每个内部子步重算。
        substeps=1
        # 世界坐标系的重力加速度 [gx, gy, gz]，单位 m/s²；当前沿 -z 方向，大小为 9.81。
        gravity=[0.0,0.0,-9.81]
        # Isaac Gym 的竖直轴枚举：0 表示 Y 轴朝上，1 表示 Z 轴朝上。
        # 此枚举不同于向量下标：z 分量的下标仍为 2；须与重力、地面和环境坐标约定一致。
        # 参考 PaiEnv 将 up_axis_idx 固定为 2，改为 Y-up 不能只修改本参数。
        up_axis=1

        class physx:
            # 以下是 PhysX 引擎参数；不会通过这个配置块自动选择引擎或启用 GPU。
            # PhysX 的 CPU 工作线程数，不是并行环境数或 GPU 线程数。
            # helpers.parse_sim_params() 中，命令行传入的正值 num_threads 会覆盖这里。
            num_threads=10
            # 约束求解器：0=PGS（Projected Gauss-Seidel），1=TGS（Temporal Gauss-Seidel）。
            # TGS 通常能更好地处理关节、堆叠等约束；这不是积分时间步数的设置。
            solver_type=1
            # 每个物理子步的位置求解迭代次数，用于减小接触穿透和关节约束误差。
            # 增大通常改善约束收敛，但计算开销也会上升；不会改变 dt 或策略频率。
            num_position_iterations=4
            # 每个物理子步的速度求解迭代次数，用于进一步修正约束相关的速度误差。
            # 0 表示不进行额外的速度迭代；位置求解阶段仍会产生冲量、改变速度，并非关闭碰撞。
            num_velocity_iterations=0
            # 接触生成偏移，单位 m；两形状间距小于双方 contact_offset 之和时生成接触候选。
            # 0.01 为单个形状的 1 cm 偏移；不表示地面会让脚始终悬空 1 cm。
            # 应大于 rest_offset；偏大增加接触数量，偏小可能使离散时间步下的接触不稳定。
            contact_offset=0.01
            # 静止接触偏移，单位 m；双方 rest_offset 之和决定静止接触时的目标表面间距。
            # 双方均为 0 时，目标是碰撞表面刚好贴合；实际仍可能存在数值误差。
            rest_offset=0.0
            # 弹性反弹的相对速度阈值，单位 m/s；接触法向接近速度低于阈值时不触发恢复性反弹。
            # 用于抑制低速接触抖动；超过阈值是否反弹，还取决于接触材质的 restitution。
            bounce_threshold_velocity=0.5
            # 求解器为消除物体相互穿透而施加的分离修正速度上限，单位 m/s。
            # 过大可能导致重叠物体被猛烈弹开，过小会使穿透恢复缓慢；不是机器人运动速度上限。
            max_depenetration_velocity=1.0
            # GPU 接触对容量上限；2**23 = 8,388,608，是整个仿真场景的容量配置。
            # 更多并行环境或更复杂碰撞可能需要更大容量；增大会占用更多显存，不代表提高精度。
            max_gpu_contact_pairs=2**23
            # PhysX 默认内部缓冲区的容量倍率（无量纲）；5 表示按默认容量的 5 倍配置。
            # 为复杂场景提供更多缓冲空间，会增加内存/显存开销；不是仿真速度或迭代次数倍率。
            default_buffer_size_multiplier=5
            # 接触信息收集模式：0=不收集，1=只收集最后一个物理子步，2=收集所有物理子步。
            # 影响接触力张量等输出，进而影响依赖它们的奖励和终止判断；0 不会关闭碰撞响应。
            # 这里的子步指 sim.substeps，不是 control.decimation；substeps=1 时模式 1、2 等价。
            contact_collection=2

class LeggedRobotCfgPPO(BaseConfig):
    seed=114514
    runner_class_name="OnPolicyRunner"
    class policy:
        init_noise_std=1.0
        actor_hidden_dims=[512,256,128]
        critic_hidden_dims=[512,256,128]

    class algorithm:
        value_loss_coef=1.0
        use_clipped_value_loss=True
        clip_param=0.2
        entropy_coef=0.01
        num_learning_epochs=5
        num_mini_batches=4
        learning_rate=1e-3
        schedule='adaptive'
        gamma=0.99
        lam=0.95
        desired_kl=0.01
        max_grad_norm=1

    class runner:
        policy_class_name="ActorCritic"
        algorithm_class_name="PPO"
        num_steps_per_env=24
        max_iterations=1500

        save_interval=1000
        experiment_name='test'
        run_name=''
        resume=False
        load_run=-1
        checkpoint=-1
        resume_path=None
