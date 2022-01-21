from typing import Union

import gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import wandb

from core.classification import ClassificationModel
from core.ner import NERModel
from utils.networks import PositionalEncoder, MLP
from utils.render import plot_ner_output

extractors = {}


def segment_ner_probs(ner_probs:Union[torch.Tensor, np.ndarray], max_segments=32):
    ner_probs = torch.tensor(ner_probs)
    if len(ner_probs.size()) == 2:
        ner_probs = ner_probs.unsqueeze(0)
    num_words = ner_probs.size(1)
    threshold, _ = torch.kthvalue(ner_probs, num_words - max_segments + 1, dim=1)
    threshold = threshold[0,0]
    segments = ner_probs[0,:,0] > threshold
    segments = segments.tolist()
    segment_data = []
    cur_seg_data = []

    def concat_seg_data(seg_data):
        seg_len = len(seg_data)
        start_prob = seg_data[0][0,0]
        seg_data = torch.cat(seg_data, dim=0)
        seg_data = torch.sum(seg_data, dim=0, keepdim=True) / seg_len
        seg_data[:,0] = start_prob
        seg_data = torch.cat((seg_data, torch.tensor(seg_len).reshape(1,1)), dim=-1)
        return seg_data

    for div_idx, divider in enumerate(segments):
        if ner_probs[0, div_idx, 0].item() == -1:
            break
        if divider and cur_seg_data:
            cur_seg_data = concat_seg_data(cur_seg_data)
            segment_data.append(cur_seg_data)
            cur_seg_data = []
        cur_slice = ner_probs[:,div_idx]
        cur_seg_data.append(cur_slice)
    if cur_seg_data:
        cur_seg_data = concat_seg_data(cur_seg_data)
        segment_data.append(cur_seg_data)
    n_segments = len(segment_data)
    segmented = torch.cat(segment_data, dim=0)
    padding = max(0, max_segments - n_segments)
    segmented = torch.cat((segmented[:max_segments], -torch.ones((padding, 10))), dim=0)
    segmented = segmented.unsqueeze(0)
    return segmented

    

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
                                                   nhead=args.env.n_head,
                                                   batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer, args.env.num_attention_layers).to(device)
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



@register_extractor
class SeqwiseFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, args) -> None:
        feature_dim = args.ner.essay_max_tokens * 10 + args.kls.max_discourse_elements * 8
        super().__init__(observation_space, features_dim=feature_dim)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.extractors = {}
        for key, _subspace in observation_space.spaces.items():
            if key == 'ner_probs':
                self.extractors[key] = nn.Flatten()
            elif key == 'seg_tokens':
                self.extractors[key] = nn.Flatten()
            elif key == 'class_tokens':
                self.extractors[key] = nn.Flatten()

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            subspace_output = extractor(observations[key])
            encoded_tensor_list.append(subspace_output)
        output = torch.cat(encoded_tensor_list, dim=-1)
        return output


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
            batch_size=seg_args.batch_size,
            learning_rate=seg_args.lr,
            n_epochs=seg_args.epochs
        ))
    return agent_cls("MultiInputPolicy", env, **agent_kwargs)
