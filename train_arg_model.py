import os

import numpy as np
import random
import transformers
import torch

from core.argument_model import ArgumentModel
from core.dataset import EssayDataset

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

    if not args.debug:
        dataset = EssayDataset()
    else:
        dataset = EssayDataset(n_essays=200)
        args.print_interval = 10
        args.eval_interval = 50
        args.batches_per_eval = 4
        args.epochs = 20

    train, val = dataset.split()
    class_train_dataset = train.make_arg_classification_dataset(balanced=True)
    class_val_dataset = val.make_arg_classification_dataset(balanced=True)
    polarity_train_dataset = train.make_polarity_dataset()
    polarity_val_dataset = val.make_polarity_dataset()

    arg_model = ArgumentModel()
    
    with wandb_run(args):
        arg_model.train(class_train_dataset, class_val_dataset,
                        polarity_train_dataset, polarity_val_dataset,
                        args)
