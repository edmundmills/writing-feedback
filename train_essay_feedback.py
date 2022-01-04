import os

import numpy as np
import random
import transformers
import torch

from core.models.essay_feedback import EssayModel
from core.dataset import EssayDataset

from utils.config import parse_args, get_config, WandBRun

if __name__ == '__main__':
    args = parse_args()
    args = get_config('train_essay_feedback', args)

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

    essay_model = EssayModel()
    
    with WandBRun(args):
        essay_model.train(train, val, args)