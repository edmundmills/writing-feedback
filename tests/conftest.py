import random
from typing import List

import numpy as np
from omegaconf import OmegaConf
import pickle
import pytest
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.env import AssignmentEnv
from core.essay import Prediction
from core.ner import NERTokenizer
from utils.config import get_config
from utils.constants import *


max_d_elems = 32
encoded_sentence_length = 768
encoded_essay_length = 1024

# ARGS

@pytest.fixture
def base_args():
    args = get_config('base')
    args.wandb = False
    return args

@pytest.fixture
def pred_args():
    args = OmegaConf.load('conf/predict/attention.yaml')
    return args

@pytest.fixture
def seg_args():
    args = OmegaConf.load('conf/seg/ppo.yaml')
    return args

@pytest.fixture
def ner_args():
    args = OmegaConf.load('conf/ner/ner.yaml')
    return args

# DATASET

@pytest.fixture
def dataset():
    return EssayDataset(n_essays=10)

@pytest.fixture
def essay():
    return EssayDataset(n_essays=1)[0]

@pytest.fixture
def pstrings():
    return ['0 1 2', '3 4 5', '6 7 8 9']

@pytest.fixture
def prediction():
    return Prediction(0, 10, 1, 1)

# ENCODERS AND TOKENIZERS

class DElemEncoder:
    def encode(self, sentences: List[str]):
        return torch.rand(max_d_elems, encoded_sentence_length + 1)


class NERModel:
    def __init__(self, *args, **kwargs) -> None:
        self.seg_only = False
    
    def inference(self, *args, **kwargs):
        return torch.ones(1, encoded_essay_length, 15).float()

@pytest.fixture
def ner_probs():
    dataset = EssayDataset.load('data/dataset_with_ner.pkl')
    return next(iter(dataset.ner_probs.values()))

@pytest.fixture
def ner_tokenizer():
    ner_args = get_config('base').ner
    return NERTokenizer(ner_args)

@pytest.fixture
def ner_model():
    return NERModel()


@pytest.fixture
def encoded_essay():
    ner_args = get_config('base').ner
    tokenizer = NERTokenizer(ner_args)
    essay = EssayDataset(1)[0]
    return tokenizer.encode(essay.text)


@pytest.fixture
def d_elem_encoder():
    return DElemEncoder()


@pytest.fixture
def encoded_preds():
    preds = [0]*500 + [-1]* (encoded_essay_length - 500)
    preds[0] = 1
    preds[20] = 1
    preds[60] = 1
    preds[90] = 1
    preds[100] = 1
    return torch.LongTensor(preds).unsqueeze(0)


# ENVS

@pytest.fixture
def dataset_with_ner_probs():
    dataset = EssayDataset(n_essays=5)
    with open('data/dataset_with_ner.pkl', 'rb') as saved_file:
        full_dataset = pickle.load(saved_file)
    dataset.ner_probs = full_dataset.ner_probs
    return dataset

@pytest.fixture
def assign_args():
    args = get_config('base', args=['env=assignment'])
    env_args = args.env
    return env_args

@pytest.fixture
def assign_env():
    args = get_config('base', args=['env=assignment'])
    dataset = EssayDataset(n_essays=5)
    with open('data/dataset_with_ner.pkl', 'rb') as saved_file:
        full_dataset = pickle.load(saved_file)
    dataset.ner_probs = full_dataset.ner_probs
    env = AssignmentEnv(dataset, args.env)
    return env


