import copy
import math

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Sliding-window GRU actor with an MLP critic."""

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        frame_stack,
        num_single_obs,
        actor_hidden_dims,
        critic_hidden_dims,
        actor_obs_embedding_dim,
        gru_hidden_size,
        gru_num_layers,
        gru_dropout,
        init_noise_std,
        activation=nn.ELU(),
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments! "
                "They will be ignored! " + str(list(kwargs.keys()))
            )

        super().__init__()

        expected_actor_obs = frame_stack * num_single_obs
        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                "Actor observation size mismatch: "
                f"num_actor_obs={num_actor_obs}, but frame_stack * "
                f"num_single_obs={frame_stack} * {num_single_obs} "
                f"= {expected_actor_obs}."
            )
        if gru_num_layers < 1:
            raise ValueError(f"gru_num_layers must be at least 1, got {gru_num_layers}.")
        if not actor_hidden_dims:
            raise ValueError("actor_hidden_dims must contain at least one hidden size.")
        if not critic_hidden_dims:
            raise ValueError("critic_hidden_dims must contain at least one hidden size.")

        self.num_actor_obs = num_actor_obs
        self.frame_stack = frame_stack
        self.num_single_obs = num_single_obs

        frame_encoder = nn.Sequential(
            nn.Linear(num_single_obs, actor_obs_embedding_dim),
            nn.LayerNorm(actor_obs_embedding_dim),
            copy.deepcopy(activation),
        )
        gru = nn.GRU(
            input_size=actor_obs_embedding_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
        )

        # Fuse the temporal summary with a direct path from the latest frame.
        actor_layers = []
        actor_input_dim = gru_hidden_size + actor_obs_embedding_dim
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(copy.deepcopy(activation))
        for layer_index, hidden_dim in enumerate(actor_hidden_dims):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dim, num_actions))
            else:
                actor_layers.append(
                    nn.Linear(hidden_dim, actor_hidden_dims[layer_index + 1])
                )
                actor_layers.append(copy.deepcopy(activation))

        self.actor = nn.ModuleDict(
            {
                "frame_encoder": frame_encoder,
                "gru": gru,
                "output_head": nn.Sequential(*actor_layers),
            }
        )

        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(copy.deepcopy(activation))
        for layer_index, hidden_dim in enumerate(critic_hidden_dims):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dim, 1))
            else:
                critic_layers.append(
                    nn.Linear(hidden_dim, critic_hidden_dims[layer_index + 1])
                )
                critic_layers.append(copy.deepcopy(activation))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Model: {self.actor}")
        print(f"Critic Model: {self.critic}")

        self.log_std = nn.Parameter(
            torch.full((num_actions,), math.log(init_noise_std))
        )
        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[index])
            for index, module in enumerate(
                module for module in sequential if isinstance(module, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        # State is recomputed from the complete observation window each call.
        pass

    def forward(self, observations):
        return self._actor_forward(observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_forward(self, observations):
        if not torch.jit.is_tracing() and (
            observations.ndim != 2 or observations.shape[-1] != self.num_actor_obs
        ):
            raise ValueError(
                "Actor observations must have shape "
                f"[batch, {self.num_actor_obs}], got {tuple(observations.shape)}."
            )

        frames = observations.reshape(
            observations.shape[0], self.frame_stack, self.num_single_obs
        )
        frame_embeddings = self.actor["frame_encoder"](frames)
        gru_output, _ = self.actor["gru"](frame_embeddings)

        temporal_feature = gru_output[:, -1, :]
        current_feature = frame_embeddings[:, -1, :]
        actor_feature = torch.cat((temporal_feature, current_feature), dim=-1)
        return self.actor["output_head"](actor_feature)

    def update_distribution(self, observations):
        mean = self._actor_forward(observations)
        self.distribution = Normal(mean, self.log_std.exp())

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self._actor_forward(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
