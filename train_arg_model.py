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

    dataset = ArgumentDataset()
    class_train_dataset, class_val_dataset = dataset.make_arg_classification_datasets()
    polarity_train_dataset, polarity_val_dataset = dataset.make_polarity_dataset(n_essays=500)
    arg_model = ArgumentModel()
    with wandb_run(args):
        arg_model.train(class_train_dataset, class_val_dataset,
                        polarity_train_dataset, polarity_val_dataset,
                        args)
