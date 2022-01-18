import os

import numpy as np
import random
import transformers
import torch

from core.ner import NERModel, NERTokenizer
from core.dataset import EssayDataset

from utils.config import parse_args, get_config, WandBRun

if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.debug:
        dataset = EssayDataset(n_essays=200)
        args.ner.print_interval = 10
        args.ner.eval_interval = 10
        args.ner.eval_samples = 10
        args.ner.epochs = 3
    else:
        dataset = EssayDataset()

    train, val = dataset.split()

    tokenizer = NERTokenizer(args.ner)

    train = train.make_ner_dataset(tokenizer, seg_only=args.ner.segmentation_only)
    val = val.make_ner_dataset(tokenizer, seg_only=args.ner.segmentation_only)

    model = NERModel(args.ner)

    with WandBRun(args, project_name='ner'):
        model.train_ner(train, val, args)
