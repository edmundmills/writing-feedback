import random
from typing import List

import numpy as np
import pytest
import torch

from core.dataset import EssayDataset
from core.env import AssigmentEnv

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


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
def sentence_encoder():
    class TestEncoder:
        def encode(self, sentences: List[str]):
            return torch.rand(32, 512)
    return TestEncoder()


