import random
from typing import List

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from core.dataset import EssayDataset
from core.env import AssigmentEnv
from core.models.essay_feedback import EssayModel

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class TestEncoder:
    def encode(self, sentences: List[str]):
        return torch.rand(len(sentences), 768)

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
def d_elem_encoder():
    return TestEncoder()

@pytest.fixture
def essay_feedback_args():
    args = OmegaConf.load('config/train_essay_feedback.yaml')
    return args

@pytest.fixture
def essay_model():
    args = OmegaConf.load('config/train_essay_feedback.yaml')
    args.num_encoder_layers = 1
    return EssayModel(args, d_elem_encoder=TestEncoder())
