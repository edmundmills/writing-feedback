import random

import numpy as np
import pytest
import torch

from core.dataset import ArgumentDataset
from core.env import AssigmentEnv

@pytest.fixture
def fix_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

@pytest.fixture
def dataset():
    return ArgumentDataset(nrows=100)

@pytest.fixture
def env():
    return AssigmentEnv(dataset_rows=100)
