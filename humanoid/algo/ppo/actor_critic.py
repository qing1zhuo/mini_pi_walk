import copy
import torch
from torch import nn
from torch.distributions import Normal
import math
class ActorCritic(nn.Module):
    def __init__(
        self,
        # 观察与动作维度
        num_actor_obs,
        num_critic_obs,
        num_actions,
        frame_stack,
        num_single_obs,
        # 模型架构参数
        actor_hidden_dims,
        critic_hidden_dims,
        transformer_dim,
        transformer_heads,
        transformer_layers,
        transformer_ff_dim,
        transformer_dropout,
        # 动作初始噪声
        init_noise_std,
        # 激活函数
        activation=nn.ELU(),
        **kwargs
    ):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments! They will be ignored!"+ str([key for key in kwargs.keys()]))

        super().__init__()

        self.num_actor_obs=num_actor_obs
        self.frame_stack=frame_stack
        self.num_single_obs=num_single_obs
        self.transformer_dim=transformer_dim
        self.transformer_heads=transformer_heads
        self.transformer_head_dim=transformer_dim//transformer_heads
        self.attention_scale=self.transformer_head_dim**(-0.5)

        # actor encoder
        frame_encoder=nn.Sequential(
            nn.Linear(num_single_obs,transformer_dim),
            nn.LayerNorm(transformer_dim)
        )
        position_embedding=nn.Embedding(frame_stack,transformer_dim)
        nn.init.normal_(position_embedding.weight,mean=0.0,std=0.02)

        # actor transformer block
        transformer_blocks=nn.ModuleList()
        for _ in range(transformer_layers):
            transformer_blocks.append(
                nn.ModuleDict(
                    {
                        "norm1":nn.LayerNorm(transformer_dim),
                        "qkv":nn.Linear(transformer_dim,3*transformer_dim),
                        "attention_output":nn.Linear(transformer_dim,transformer_dim),  # 注意力后加个投影层再接ffn
                        "norm2":nn.LayerNorm(transformer_dim),
                        "ffn":nn.Sequential(
                            nn.Linear(transformer_dim,transformer_ff_dim),
                            nn.ELU(),
                            nn.Linear(transformer_ff_dim,transformer_dim),
                        ),
                    }
                )
            )
        # actor mlp
        actor_layers=[]
        actor_input_dim=2*transformer_dim
        actor_layers.append(nn.Linear(actor_input_dim,actor_hidden_dims[0]))
        actor_layers.append(copy.deepcopy(activation))
        for l in range(len(actor_hidden_dims)):
            if l==len(actor_hidden_dims)-1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l],actor_hidden_dims[l+1]))
                actor_layers.append(copy.deepcopy(activation))
        # actor model
        self.actor=nn.ModuleDict(
            {
                "frame_encoder":frame_encoder,
                "position_embedding":position_embedding,
                "transformer_blocks":transformer_blocks,
                "output_head":nn.Sequential(*actor_layers)
            }
        )

        # critic model
        critic_layers=[]
        critic_layers.append(nn.Linear(num_critic_obs,critic_hidden_dims[0]))
        critic_layers.append(copy.deepcopy(activation))
        for l in range(len(critic_hidden_dims)):
            if l==len(critic_hidden_dims)-1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l],1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l],critic_hidden_dims[l+1]))
                critic_layers.append(copy.deepcopy(activation))
        self.critic=nn.Sequential(*critic_layers)

        print(f"Actor Model: {self.actor}")
        print(f"Critic Model: {self.critic}")

        self.log_std=nn.Parameter(
            torch.full((num_actions,),math.log(init_noise_std)) # TODO PR 所有log_std都要
        )
        self.distribution=None
        Normal.set_default_validate_args=False
    
    @staticmethod
    def init_weights(sequential,scales):
        [torch.nn.init.orthogonal_(module.weight,gain=scales[idx]) for idx, module in enumerate(mod for mod in sequential if isinstance(mod,nn.Linear))]

    def reset(self,dones=None):
        pass

    def forward(self,observations):
        return self._actor_forward(observations)

    @property   # 使用property当属性直接访问
    def action_mean(self):
        return self.distribution.mean
    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def _actor_forward(self,observations):
        batch_size=observations.shape[0]
        frames=observations.reshape(
            batch_size,
            self.frame_stack,
            self.num_single_obs,
        )# (B,S,D)

        # encode
        frame_embeddings=self.actor["frame_encoder"](frames)
        position_embeddings=self.actor["position_embedding"].weight.unsqueeze(0)
        x=frame_embeddings+position_embeddings  # (B,S,D)

        # attention

        for block in self.actor["transformer_blocks"]:
            normalized_x=block["norm1"](x)
            query,key,value=torch.chunk(block["qkv"](normalized_x),3,dim=-1)
            query=query.reshape(
                batch_size,
                self.frame_stack,
                self.transformer_heads,
                self.transformer_head_dim,
            ).transpose(1,2)
            key=key.reshape(
                batch_size,
                self.frame_stack,
                self.transformer_heads,
                self.transformer_head_dim,
            ).transpose(1,2)
            value=value.reshape(
                batch_size,
                self.frame_stack,
                self.transformer_heads,
                self.transformer_head_dim,
            ).transpose(1,2)

            attention_scores=torch.matmul(query,key.transpose(-2,-1))
            attention_scores=attention_scores*self.attention_scale
            attention_weights=torch.softmax(attention_scores,dim=-1)
            attention_output=torch.matmul(attention_weights,value)  #(B,H,S,D)
            attention_output=attention_output.transpose(1,2).contiguous().reshape(batch_size,self.frame_stack,self.transformer_dim)

            x=x+block["attention_output"](attention_output)
            x=x+block["ffn"](block["norm2"](x))

        # 提取最后一帧特征
        temporal_feature=x[:,-1,:]
        current_feature=frame_embeddings[:,-1,:]
        actor_feature=torch.cat((temporal_feature,current_feature),dim=-1)
        return self.actor["output_head"](actor_feature)

    def update_distribution(self,observations):
        mean=self._actor_forward(observations)
        self.distribution=Normal(mean,self.log_std.exp())
    
    def act(self,observations,**kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self,actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self,observations):
        actions_mean=self._actor_forward(observations)
        return actions_mean
    
    def evaluate(self,critic_observations,**kwargs):
        value=self.critic(critic_observations)
        return value