import os

import numpy as np
import random
import transformers
import torch
import wandb

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.ner import NERModel, NERTokenizer
from core.dataset import EssayDataset
from utils.config import parse_args, get_config, WandBRun


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    if args.debug:
        dataset = EssayDataset(n_essays=100)
        dataset.make_folds()
        args.ner.print_interval = 10
        args.ner.eval_interval = 10
        args.ner.eval_samples = 10
        args.ner.epochs = 2
    else:
        dataset = EssayDataset.load(args.dataset_path)

    tokenizer = NERTokenizer(args.ner)

    first_run_name = None
    
    for fold in dataset.folds:
        print(f'Starting training on fold {fold}')
        train, val = dataset.get_fold(fold)

        train = tokenizer.make_ner_dataset(train)
        val = tokenizer.make_ner_dataset(val)
        print(f'Training dataset size: {len(train)}')
        print(f'Validation dataset size: {len(val)}')

        model = NERModel(args.ner)

        with WandBRun(args, project_name='ner'):
            run_name = wandb.run.name if args.wandb else 'test'
            first_run_name = first_run_name or run_name
            model.train_ner(train, val, args)
            if args.ner.save_model:
                model_name = f'{first_run_name}_fold_{fold}_{wandb.run.name}' if args.wandb else 'test'
                model.save(model_name)
