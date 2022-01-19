import random
from typing import List

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from core.d_elems import DElemTokenizer
from core.dataset import EssayDataset
from core.env import AssigmentEnv, DividerEnv, SequencewiseEnv, SplitterEnv, WordwiseEnv
from core.essay import Prediction
from core.classification import ClassificationModel
from utils.config import get_config

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

max_d_elems = 32
encoded_sentence_length = 768
encoded_essay_length = 1024


@pytest.fixture
def fix_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

# ARGS

@pytest.fixture
def base_args():
    args = get_config('base')
    args.wandb = False
    return args

@pytest.fixture
def kls_args():
    args = OmegaConf.load('conf/kls/classification.yaml')
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


class NERTokenized:
    def __init__(self, input_ids, attention_mask) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, attribute):
        if attribute == 'input_ids':
            return self.input_ids
        elif attribute == 'attention_mask':
            return self.attention_mask
        else:
            raise KeyError()
    
    def word_ids(self):
        return list(range(len(self.input_ids)))


class NERTokenizer:
    def __init__(self) -> None:
        self.max_tokens = encoded_essay_length
        
    def encode(self, text:str):
        encoded_text = torch.LongTensor(list(range(encoded_essay_length))).unsqueeze(0)
        attention_mask = torch.ones(1, encoded_essay_length, dtype=torch.uint8)
        word_ids = list(range(-1, encoded_text.size(1) - 1))
        word_id_tensor = torch.LongTensor(
            [word_id if word_id is not None else -1 for word_id in word_ids]
        ).unsqueeze(0)
        return {'word_ids': word_ids, 'input_ids': encoded_text,
                'attention_mask': attention_mask, 'word_id_tensor': word_id_tensor}

@pytest.fixture
def d_elem_tokenizer():
    kls_args = get_config('base').kls
    d_elem_tokenizer = DElemTokenizer(kls_args)
    return d_elem_tokenizer

@pytest.fixture
def ner_tokenizer():
    return NERTokenizer()

@pytest.fixture
def encoded_essay():
    return NERTokenizer().encode('')


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


# MODELS

@pytest.fixture
def kls_model():
    args = get_config('base')
    args.kls.num_attention_layers = 1
    return ClassificationModel(args.kls, d_elem_encoder=DElemEncoder())

# ENVS

@pytest.fixture
def assign_env():
    return AssigmentEnv(n_essays=10)

@pytest.fixture
def splitter_args():
    args = get_config('base', args=['env=splitter'])
    env_args = args.env
    return env_args

@pytest.fixture
def splitter_env():
    dataset = EssayDataset(n_essays=10)
    encoder = NERTokenizer()
    args = get_config('base', args=['env=splitter'])
    args.kls.num_attention_layers = 1
    d_elem_tokenizer = DElemTokenizer(args.kls)
    env = SplitterEnv(dataset, encoder, d_elem_tokenizer, args.env)
    return env

@pytest.fixture
def seqwise_args():
    args = get_config('base', args=['env=seqwise'])
    env_args = args.env
    return env_args

@pytest.fixture
def seq_env():
    dataset = EssayDataset(n_essays=10)
    encoder = NERTokenizer()
    args = get_config('base', args=['env=seqwise'])
    args.kls.num_attention_layers = 1
    d_elem_tokenizer = DElemTokenizer(args.kls)
    env = SequencewiseEnv(dataset, encoder, d_elem_tokenizer, args.env)
    return env

@pytest.fixture
def divider_args():
    args = get_config('base', args=['env=divider'])
    env_args = args.env
    return env_args

@pytest.fixture
def divider_env():
    dataset = EssayDataset(n_essays=10)
    encoder = NERTokenizer()
    args = get_config('base', args=['env=divider'])
    args.kls.num_attention_layers = 1
    d_elem_tokenizer = DElemTokenizer(args.kls)
    env = DividerEnv(dataset, encoder, d_elem_tokenizer, args.env)
    return env

@pytest.fixture
def wordwise_args():
    args = get_config('base', args=['env=wordwise'])
    env_args = args.env
    return env_args

@pytest.fixture
def word_env():
    args = get_config('base', args=['env=wordwise'])
    dataset = EssayDataset(n_essays=10)
    encoder = NERTokenizer()
    args.kls.num_attention_layers = 1
    d_elem_tokenizer = DElemTokenizer(args.kls)
    env = WordwiseEnv(dataset, encoder, d_elem_tokenizer, args.env)
    return env


