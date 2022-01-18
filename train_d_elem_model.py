import os

import numpy as np
import random
import transformers
import torch

from core.d_elems import DElemModel
from core.dataset import EssayDataset

from utils.config import parse_args, get_config, WandBRun

if __name__ == '__main__':
    args = parse_args()
    args = get_config('arg_model', args)

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

    arg_model = DElemModel()
    
    with WandBRun(args, project_name='d_elelm_model'):
        arg_model.train(class_train_dataset, class_val_dataset,
                        polarity_train_dataset, polarity_val_dataset,
                        args)
