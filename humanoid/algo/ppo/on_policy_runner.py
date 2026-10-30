import os
import time
import torch
import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo import PPO
from .actor_critic import ActorCritic
from humanoid.algo.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

class OnPolicyRunner:
    def __init__(
        self,
        env:VecEnv,
        train_cfg,
        log_dir=None,
        device="cpu"
    ):
        self.cfg=train_cfg["runner"]
        self.alg_cfg=train_cfg["algorithm"]
        self.policy_cfg=train_cfg["policy"]
        self.all_cfg=train_cfg

        self.wandb_run_name=(
            datetime.now().strftime("%b%d_%H-%M-%S")+"_"+
            train_cfg["runner"]["experiment_name"]+"_"+
            train_cfg["runner"]["run_name"]
        )

        self.device=device
        self.env=env
        if self.env.num_privileged_obs is not None:
            num_critic_obs=self.env.num_privileged_obs
        else:
            num_critic_obs=self.env.num_obs

        # 创建AC policy模型类
        actor_critic_class=eval(self.cfg["policy_class_name"])
        actor_critic:ActorCritic=actor_critic_class(
            self.env.num_obs,num_critic_obs,self.env.num_actions,**self.policy_cfg
        ).to(self.device)

        # 创建PPO alg算法类
        alg_class=eval(self.cfg["algorithm_class_name"])
        self.alg:PPO=alg_class(actor_critic,device=self.device,**self.alg_cfg)

        # 配置runner算法类
        self.num_steps_per_env=self.cfg["num_steps_per_env"]
        self.save_interval=self.cfg["save_interval"]

        # 初始化storage和model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions]
        )

        # log
        self.log_dir=log_dir
        self.writer=None
        self.tot_timesteps=0
        self.tot_time=0
        self.current_learning_iteration=0

        self.env.reset()
    
    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        '''
        参数列表：
            当前要学习多少轮
            初始是否从随机时间点开始
        '''
        # 初始化日志writer
        if self.log_dir is not None and self.writer is None:
            wandb.init(
                project="pai_ppo",
                sync_tensorboard=True,
                name=self.wandb_run_name,
                config=self.all_cfg
            )
            self.writer=SummaryWriter(log_dir=self.log_dir,flush_secs=10)

        # 生成随机时间点开始的长度buffer
        if init_at_random_ep_len:
            self.env.episode_length_buf=torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        # obs
        obs=self.env.get_observations()
        privileged_obs=self.env.get_privileged_observations()
        critic_obs=privileged_obs if privileged_obs is not None else obs
        obs,critic_obs=obs.to(self.device),critic_obs.to(self.device)

        self.alg.actor_critic.train()

        # 创建buffer
        ep_infos=[]     # episode的信息buffer
        rewbuffer=deque(maxlen=100) # reward buffer
        lenbuffer=deque(maxlen=100) # length buffer
        cur_reward_sum=torch.zeros( # 当前奖励综合buffer
            self.env.num_envs,
            dtype=torch.float,
            device=self.device
        )
        cur_episode_length=torch.zeros( # 当前episode的长度
            self.env.num_envs,
            dtype=torch.float,
            device=self.device
        )

        tot_iter=self.current_learning_iteration+num_learning_iterations
        for it in range(self.current_learning_iteration,tot_iter):
            start=time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 根据当前观察获取动作，并根据动作更新环境
                    actions=self.alg.act(obs,critic_obs)
                    obs,privileged_obs,rewards,dones,infos=self.env.step(actions)
                    critic_obs=privileged_obs if privileged_obs is not None else obs
                    obs,critic_obs,rewards,dones=(
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device)
                    )
                    # 算法依据反馈添加transition
                    self.alg.process_env_step(rewards,dones,infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum+=rewards
                        cur_episode_length+=1
                        # 找出已经结束的环境，并记录这些结束的环境的buffer
                        new_ids=(dones>0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:,0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:,0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids]=0
                        cur_episode_length[new_ids]=0
                stop=time.time()
                collection_time=stop-start

                # learning step
                start=stop
                # 算法依据最后一刻观测计算总体reward
                self.alg.compute_returns(critic_obs)

            # 算法依据buffer更新模型
            mean_value_loss,mean_surrogate_loss=self.alg.update()
            stop=time.time()
            learn_time=stop-start

            # 记录已完成的更新次数；恢复训练时从下一轮继续。
            self.current_learning_iteration=it+1
            if self.log_dir is not None:
                self.log(locals())
            if self.current_learning_iteration%self.save_interval==0:
                self.save(os.path.join(self.log_dir,"model_{}.pt".format(self.current_learning_iteration)))
            ep_infos.clear()

        self.save(
            os.path.join(
                self.log_dir,"model_{}.pt".format(self.current_learning_iteration)
            )
        )
    
    def log(self,locs,width=80,pad=50):
        # 汇总本轮训练指标，写入 TensorBoard，并在终端打印训练进度。
        # 参数列表：
        #   locs：learn() 传入的 locals() 字典，包含本轮采样/学习耗时、
        #         ep_infos、损失均值、奖励/回合长度缓冲区、迭代编号 it
        #         以及本次计划训练的轮数 num_learning_iterations。
        #   width：终端日志的分隔线宽度和标题居中宽度。
        #   pad：终端日志中指标名称右对齐的字段宽度。
        # 调用前需初始化 self.writer；函数更新累计步数和耗时，无返回值。

        # 一轮采样步数等于每个环境的采样步数乘以并行环境数；
        # 累计耗时仅统计采样和参数更新，不包含日志写入、模型保存等开销。
        self.tot_timesteps+=self.num_steps_per_env*self.env.num_envs
        self.tot_time+=locs["collection_time"]+locs["learn_time"]
        iteration_time=locs["collection_time"]+locs["learn_time"]

        ep_string=f""

        if locs["ep_infos"]:
            # 以首条 episode 信息中的指标为准，要求各条信息包含相同的键。
            for key in locs["ep_infos"][0]:
                infotensor=torch.tensor([],device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 将数值和零维张量转换为可拼接的张量；这里会更新 ep_info。
                    if not isinstance(ep_info[key],torch.Tensor):
                        ep_info[key]=torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape)==0:
                        ep_info[key]=ep_info[key].unsqueeze(0)
                    # 统一设备后汇总该指标，均值按拼接后的所有元素计算。
                    infotensor=torch.cat((infotensor,ep_info[key].to(self.device)))
                value=torch.mean(infotensor)

                # TensorBoard 以训练迭代编号为横轴，终端文本保留四位小数。
                self.writer.add_scalar("Episode/"+key,value,locs["it"])
                ep_string+=f"""{f'Mean episode {key}:':>{pad}}{value:.4f}\n"""
                
        # 动作分布的平均标准差用于观察探索强度；FPS 表示所有环境合计
        # 每秒处理的采样步数，分母同时包含采样耗时和学习耗时。
        mean_std=self.alg.actor_critic.log_std.exp().mean()
        fps=int(
            self.num_steps_per_env*self.env.num_envs/(locs["collection_time"]+locs["learn_time"])
        )
        # 按迭代记录价值损失、策略代理损失、学习率、探索噪声及运行性能。
        # lr_loss 是现有日志标签，实际记录的是学习率。
        self.writer.add_scalar("value_loss",locs["mean_value_loss"],locs["it"])
        self.writer.add_scalar("surrogate_loss",locs["mean_surrogate_loss"],locs["it"])
        self.writer.add_scalar("lr_loss",self.alg.learning_rate,locs["it"])
        self.writer.add_scalar("mean_noise_std",mean_std.item(),locs["it"])
        self.writer.add_scalar("total_fps",fps,locs["it"])
        self.writer.add_scalar("collection_time",locs["collection_time"],locs["it"])
        self.writer.add_scalar("learning_time",locs["learn_time"],locs["it"])

        if len(locs["rewbuffer"])>0:
            # 两个缓冲区保存最近最多 100 个已结束回合的累计奖励和长度。
            # 分别按迭代编号和累计训练秒数记录，便于比较学习进度与耗时。
            self.writer.add_scalar("mean_reward",statistics.mean(locs["rewbuffer"]),locs["it"])
            self.writer.add_scalar("mean_episode_length",statistics.mean(locs["lenbuffer"]),locs["it"])
            self.writer.add_scalar("mean_reward_time",statistics.mean(locs["rewbuffer"]),self.tot_time)
            self.writer.add_scalar("mean_episode_length_time",statistics.mean(locs["lenbuffer"]),self.tot_time)

        # 拼接终端标题和指标；尚无回合结束时，省略奖励及回合长度均值。
        str=f" \033[1m learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033 [0m "
        if len(locs["rewbuffer"])>0:
            log_string=(
                f"""{'#'*width}\n"""
                f"""{str.center(width,' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        # 追加环境提供的 episode 指标，以及累计步数、耗时和剩余时间估计。
        # ETA 按累计平均每轮耗时乘以剩余轮数估算；当前公式假定从第 0 轮
        # 开始训练，恢复训练时 it 的偏移会影响估算结果。
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['tot_iter'] - locs['it']-1):.1f}s\n"""  #TODO PR
        )
        print(log_string)

    def save(self,path,infos=None):
        torch.save(
            {
                "model_state_dict":self.alg.actor_critic.state_dict(),
                "optimizer_state_dict":self.alg.optimizer.state_dict(),
                "iter":self.current_learning_iteration,
                "infos":infos,
            },
            path,
        )

    def load(self,path,load_optimizer=True):
        loaded_dict=torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # 自适应调度使用此字段，须与恢复后的优化器学习率一致。
            self.alg.learning_rate=self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration=loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self,device):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self,device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
