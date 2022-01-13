import random
from typing import List

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from core.dataset import EssayDataset
from core.env import AssigmentEnv, SegmentationEnv
from core.essay import Prediction
from core.models.essay_feedback import EssayModel

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

encoded_sentence_length = 768
encoded_essay_length = 1024

class TestEncoder:
    def encode(self, sentences: List[str]):
        return torch.rand(len(sentences), encoded_sentence_length)

class TestSegmentationTokenizer:
    def __init__(self) -> None:
        self.max_tokens = encoded_essay_length
        
    def encode(self, text:str):
        encoded_text = torch.LongTensor(list(range(encoded_essay_length))).unsqueeze(0)
        attention_mask = torch.ones(1, encoded_essay_length, dtype=torch.uint8)
        return {'input_ids': encoded_text, 'attention_mask': attention_mask}

@pytest.fixture
def fix_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

@pytest.fixture
def dataset():
    return EssayDataset(n_essays=10)

@pytest.fixture
def env():
    return AssigmentEnv(n_essays=10)

@pytest.fixture
def essay():
    return EssayDataset(n_essays=1)[0]

@pytest.fixture
def pstrings():
    return ['0 1 2', '3 4 5', '6 7 8 9']

@pytest.fixture
def prediction():
    return Prediction(0, 10, 1, 1)

@pytest.fixture
def d_elem_encoder():
    return TestEncoder()

@pytest.fixture
def essay_feedback_args():
    args = OmegaConf.load('config/essay_feedback.yaml')
    return args

@pytest.fixture
def essay_model():
    args = OmegaConf.load('config/essay_feedback.yaml')
    args.num_encoder_layers = 1
    return EssayModel(args, d_elem_encoder=TestEncoder())

@pytest.fixture
def seg_args():
    args = OmegaConf.load('config/segmentation.yaml')
    return args

@pytest.fixture
def seg_env():
    dataset = EssayDataset(n_essays=10)
    encoder = TestSegmentationTokenizer()
    args = OmegaConf.load('config/segmentation.yaml') 
    env = SegmentationEnv(dataset, encoder, None, args)
    return env

@pytest.fixture
def seg_tokenizer():
    return TestSegmentationTokenizer()

@pytest.fixture
def encoded_essay():
    return TestSegmentationTokenizer().encode('')

@pytest.fixture
def encoded_preds():
    preds = [0]*500 + [-1]* (encoded_essay_length - 500)
    preds[0] = 1
    preds[20] = 1
    preds[60] = 1
    preds[90] = 1
    preds[100] = 1
    return torch.LongTensor(preds).unsqueeze(0)

