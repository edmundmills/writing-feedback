import gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import wandb

from core.d_elems import DElemEncoder
from core.classification import ClassificationModel
from core.ner import NERModel


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



class SeqwiseFeatures(BaseFeaturesExtractor):
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
            elif key == 'pred_tokens':
                self.extractors[key] = nn.Flatten()

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        output = torch.cat(encoded_tensor_list, dim=-1)
        return output




def make_agent(base_args, env):
    policy_kwargs = dict(
        features_extractor_class=SeqwiseFeatures,
        features_extractor_kwargs=dict(args=base_args),
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])]
    )

    log_dir = wandb.run.name if base_args.wandb else 'test'
    return PPO("MultiInputPolicy", env,
               policy_kwargs=policy_kwargs,
               verbose=base_args.seg.sb3_verbosity,
               tensorboard_log=f"./log/{log_dir}/")
