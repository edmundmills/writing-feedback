import os

import numpy as np
import random
import transformers
import torch

from core.models.segmentation import SegmentationModel, SegmentationTokenizer
from core.dataset import EssayDataset

from utils.config import parse_args, get_config, WandBRun

if __name__ == '__main__':
    args = parse_args()
    args = get_config('segmentation', args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.debug:
        dataset = EssayDataset(n_essays=200)
        args.print_interval = 10
        args.eval_interval = 10
        args.eval_samples = 10
        args.ner_epochs = max(args.ner_epochs, 20)
    else:
        dataset = EssayDataset()

    train, val = dataset.split()

    tokenizer = SegmentationTokenizer(args)

    train = train.make_ner_dataset(tokenizer)
    val = val.make_ner_dataset(tokenizer)

    model = SegmentationModel(args)

    with WandBRun(args):
        model.train_ner(train, val, args)
