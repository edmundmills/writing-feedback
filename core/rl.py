from typing import Tuple

import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import wandb

from utils.networks import PositionalEncoder, MLP
from utils.render import plot_ner_output

extractors = {}

def register_extractor(cls):
    extractors.update({cls.__name__: cls})
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)
    return wrapper


@register_extractor
class SegmentAttention(BaseFeaturesExtractor):
    def __init__(self, observation_space, args):
        feature_dim = args.env.num_d_elems * 9
        super().__init__(observation_space, features_dim=feature_dim)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder_layer = nn.TransformerEncoderLayer(d_model=18,
                                                   nhead=args.rl.n_head,
                                                   batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer, args.rl.num_attention_layers).to(device)
        # self.positional_encoder = PositionalEncoder(features=18, seq_len=32, device=device)
        self.linear = MLP(n_inputs=18,
                        n_outputs=9,
                        n_layers=1,
                        layer_size=None).to(device)


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #     observations = self.positional_encoder(observations)
        output = self.attention(observations)
        output = self.linear(output)
        output = output.flatten(start_dim=1)
        return output


class TransformerPolicyNet(nn.Module):
    def __init__(self,
                 feature_dim,
                 last_layer_dim_pi,
                 last_layer_dim_vf,
                 d_model=16,
                 nhead=8,
                 n_attention_layers=6,
                 n_outputs=2) -> None:
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   batch_first=True)
        self.policy_net = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, n_attention_layers),
            nn.Linear(d_model, n_outputs)
        )

        self.value_net = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, n_attention_layers),
            nn.Linear(d_model, n_outputs)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = features.reshape(-1, 40, 17)
        msk = features[...,0]
        features = features[...,1:]
        output_p = self.policy_net(features)[msk]
        output_v = self.value_net(features)[msk]
        return output_p, output_v

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(-1, 40, 17)
        msk = features[...,0]
        features = features[...,1:]
        return self.policy_net(features)[msk]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(-1, 40, 17)
        msk = features[...,0]
        features = features[...,1:]
        return self.value_net(features)[msk]






def make_agent(base_args, env):
    seg_args = base_args.rl
    extractor_class = extractors[base_args.rl.feature_extractor]
    log_dir = wandb.run.name if base_args.wandb else 'test'
    agent_kwargs = dict(
        verbose=base_args.rl.sb3_verbosity,
        tensorboard_log=f"log/{log_dir}/",
    )
    if seg_args.name == 'sac':
        agent_cls = SAC
        policy_kwargs = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(args=base_args),
            activation_fn=nn.ReLU,
            net_arch=[seg_args.layer_size]*seg_args.n_layers,
        )
        agent_kwargs.update(dict(
            policy_kwargs=policy_kwargs,
            batch_size=seg_args.batch_size
        ))
    elif seg_args.name == 'ppo':
        policy_kwargs = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(args=base_args),
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[seg_args.layer_size]*seg_args.n_layers,
                        vf=[seg_args.layer_size]*seg_args.n_layers)]
        )
        agent_cls = PPO
        agent_kwargs.update(dict(
            policy_kwargs=policy_kwargs,
            n_steps=seg_args.n_steps,
            batch_size=seg_args.batch_size,
            learning_rate=seg_args.lr,
            n_epochs=seg_args.epochs
        ))
    return agent_cls("MultiInputPolicy", env, **agent_kwargs)
