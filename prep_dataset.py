import os
from tkinter import N

import numpy as np
import pickle
import random
import transformers
import torch
import wandb

from core.dataset import EssayDataset
from core.ner import NERModel, NERTokenizer
from utils.config import parse_args, get_config
from utils.constants import ner_probs_path


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = EssayDataset()
    dataset.make_folds(num_folds=5)
    dataset.save('data/full_dataset.pkl')
