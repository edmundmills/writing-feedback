import os

import numpy as np
from pathlib import Path
import random
import transformers
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.ner import NERModel, NERTokenizer
from utils.config import parse_args, get_config
from utils.constants import ner_probs_path


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    dataset = EssayDataset.load(args.dataset_path)
    
    tokenizer = NERTokenizer(args.ner)
    
    for fold in dataset.folds:
        print(f'Starting inference on fold {fold}')
        _train, val = dataset.get_fold(fold)

        ner_model = NERModel(args.ner)
        ner_model_name = f'{args.ner_model_name}_fold_{fold}'
        model_dir = Path(f'./models/{ner_model.__class__.__name__}')
        model_file = next(model_dir.glob(f'*{ner_model_name}*')).stem
        ner_model.load(model_file)

        ner_probs = ner_model.infer_for_dataset(val, tokenizer)
        dataset.ner_probs.update(ner_probs)

    dataset.save(args.ner_dataset_path)

