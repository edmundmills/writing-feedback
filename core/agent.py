from typing import Tuple, Callable, Union, List, Optional, Dict, Type

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import wandb

from core.prediction import Predicter
from utils.render import plot_ner_output



class TransformerPolicyNet(nn.Module):
    def __init__(self,
                 feature_dim,
                 args,
                 last_layer_dim_pi=15,
                 last_layer_dim_vf=15) -> None:
        super().__init__()
        self.args = args
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = Predicter(args.predict)
        self.value_net = Predicter(args.predict)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = features.reshape(-1, 40, 17)
        msk = features[...,-1].bool()
        features = features[...,:-1]
        output_p = self.policy_net(features)
        output_p = output_p[msk]
        output_v = self.value_net(features)[msk]
        return output_p, output_v

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        # self.policy_net.load(self.args.rl.saved_model_name)
        features = features.reshape(-1, 40, 17)
        msk = features[...,-1].bool()
        features = features[...,:-1]
        return self.policy_net(features)[msk]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(-1, 40, 17)
        msk = features[...,-1].bool()
        features = features[...,:-1]
        return self.value_net(features)[msk]
    

class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = None,
        *args,
        **kwargs,
    ):
        self.args = kwargs.pop('args')
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TransformerPolicyNet(self.features_dim, self.args)

    def _build(self, *args, **kwargs) -> None:   
        super()._build(*args, **kwargs)
        self.action_net = nn.Sequential()  


def make_agent(base_args, env):
    rl_args = base_args.rl
    # extractor_class = extractors[base_args.rl.feature_extractor]
    log_dir = wandb.run.name if base_args.wandb else 'test'
    agent_kwargs = dict(
        verbose=base_args.rl.sb3_verbosity,
        tensorboard_log=f"log/{log_dir}/",
    )
    if rl_args.name == 'ppo':
        agent_cls = PPO
        agent_kwargs.update(dict(
            n_steps=rl_args.n_steps,
            batch_size=rl_args.batch_size,
            learning_rate=rl_args.lr,
            n_epochs=rl_args.epochs,
            target_kl=rl_args.target_kl,
            policy_kwargs={'args': base_args}
        ))
    agent = agent_cls(TransformerActorCriticPolicy, env, **agent_kwargs)
    agent.policy.mlp_extractor.policy_net.load(base_args.rl.saved_model_name)
    agent.policy.mlp_extractor.value_net.load(base_args.rl.saved_model_name)
    return agent