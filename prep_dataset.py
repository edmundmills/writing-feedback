import os
from tkinter import N

import numpy as np
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


from core.dataset import EssayDataset
from utils.config import parse_args, get_config


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    dataset = EssayDataset()
    dataset.make_folds(num_folds=5)
    dataset.save('data/full_dataset.pkl')
