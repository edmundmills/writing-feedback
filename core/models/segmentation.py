import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from transformers import LongformerTokenizer, LongformerModel


class SegmentationModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.transformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 2)
        )

    def forward(self, essay_tokens):
        essay_tokens = essay_tokens.long()
        tokenized_text, attention_mask = torch.chunk(essay_tokens, 2, dim=-2)
        tokenized_text = tokenized_text.squeeze(-2)
        attention_mask = attention_mask.squeeze(-2)
        with torch.no_grad():
            encoded = self.transformer(tokenized_text, attention_mask=attention_mask)
        logits = self.classifier(encoded.last_hidden_state)
        output, _ = torch.chunk(logits, 2, dim=-1)
        return output.flatten(1)


class EssayFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, args) -> None:
        feature_dim = args.essay_max_tokens * 2
        super().__init__(observation_space, features_dim=feature_dim)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.extractors = {}
        for key, _subspace in observation_space.spaces.items():
            if key == 'essay_tokens':
                self.extractors[key] = SegmentationModel(args).to(device)
            elif key == 'pred_tokens':
                self.extractors[key] = nn.Flatten()


    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        output = torch.cat(encoded_tensor_list, dim=-1)
        return output


class SegmentationTokenizer:
    def __init__(self, args):
        super().__init__()
        # self.action_space_dim = args.action_space_dim # could possibly drop to 200
        self.max_tokens = args.essay_max_tokens
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        # self.model = SegmentationModel(args).to(self.device)

    def encode(self,text:str):
        tokenized = self.tokenizer.encode_plus(text,
                                             max_length=self.max_tokens,
                                             padding='max_length',
                                             truncation=True,
                                             return_tensors='pt',
                                             return_attention_mask=True)
        return tokenized


def make_agent(args, env):
    policy_kwargs = dict(
        features_extractor_class=EssayFeatures,
        features_extractor_kwargs=dict(args=args),
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256], vf=[256])]
    )

    return PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
