import os

import numpy as np
import random
import transformers
import torch

from core.argument_model import ArgumentModel
from core.dataset import ArgumentDataset

from utils.config import parse_args, get_config, wandb_run

if __name__ == '__main__':
    args = parse_args()
    args = get_config(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dataset, val_dataset = ArgumentDataset.create_train_test_split(fraction=0.9)
    arg_model = ArgumentModel()
    with wandb_run(args):
        arg_model.train(train_dataset, val_dataset, args)
