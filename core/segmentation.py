from git import base
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import wandb

from core.d_elems import DElemEncoder
from core.classification import ClassificationModel
from core.ner import NERModel

extractors = {}

def register_extractor(cls):
    extractors.update({cls.__name__: cls})
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)
    return wrapper


@register_extractor
class WordwiseFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, args) -> None:
        feature_dim = args.ner.essay_max_tokens * 2
        super().__init__(observation_space, features_dim=feature_dim)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.extractors = {}
        for key, _subspace in observation_space.spaces.items():
            if key == 'essay_tokens':
                ner_model = NERModel(args.ner, feature_extractor=True)
                if not args.seg.train_ner_model:
                    ner_model.load(args.seg.ner_model_name)
                ner_model = ner_model.to(device)
                self.extractors[key] = ner_model
            elif key == 'prev_d_elem_tokens':
                d_elem_classifier = ClassificationModel(args.kls).to(device)
                self.extractors[key] = d_elem_classifier


    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        output = torch.cat(encoded_tensor_list, dim=-1)
        return output


@register_extractor
class SeqwiseFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, args) -> None:
        feature_dim = args.ner.essay_max_tokens * 16
        super().__init__(observation_space, features_dim=feature_dim)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.extractors = {}
        for key, _subspace in observation_space.spaces.items():
            if key == 'essay_tokens':
                ner_model = NERModel(args.ner, feature_extractor=True)
                if not args.seg.train_ner_model:
                    ner_model.load(args.seg.ner_model_name)
                ner_model = ner_model.to(device)
                self.extractors[key] = ner_model
            elif key == 'pred_tokens':
                self.extractors[key] = nn.Flatten()

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        output = torch.cat(encoded_tensor_list, dim=-1)
        return output


def make_agent(base_args, env):
    seg_args = base_args.seg
    extractor_class = extractors[base_args.env.feature_extractor]
    log_dir = wandb.run.name if base_args.wandb else 'test'
    agent_kwargs = dict(
        verbose=base_args.seg.sb3_verbosity,
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
            batch_size=seg_args.batch_size
        ))
    return agent_cls("MultiInputPolicy", env, **agent_kwargs)
