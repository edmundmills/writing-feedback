import os

import numpy as np
import random
import torch
import transformers
import wandb

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.segment_transformer import SegmentTransformer
from core.dataset import EssayDataset
from utils.config import parse_args, get_config, WandBRun


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()

    ner_dataset = EssayDataset.load(args.segmented_dataset_path)

    if args.debug:
        dataset = EssayDataset(n_essays=20)
        dataset.copy_essays(ner_dataset)
        dataset.make_folds(2)
        args.predict.print_interval = 10
        args.predict.eval_interval = 10
        args.predict.eval_samples = 5
        args.predict.epochs = 10
    else:
        dataset = ner_dataset

    first_run_name = None

    for fold in dataset.folds:
        with WandBRun(args, project_name='seg_t'):
            print(f'Starting training on fold {fold}')
            train, val = dataset.get_fold(fold)
            model = SegmentTransformer(args.seg_t)
            run_name = wandb.run.name if args.wandb else 'test'
            first_run_name = first_run_name or run_name
            model.finetune(train, val, args.seg_t)
            if args.seg_t.save_model:
                model_name = f'{first_run_name}_fold_{fold}_{wandb.run.name}' if args.wandb else 'test'
                model.save(model_name)
            if args.seg_t.save_tokenized_dataset:
                val = model.tokenize_dataset(val)
                dataset.copy_essays(val)
    if args.seg_t.save_tokenized_dataset:
        dataset.save('data/seg_t_tokenized_dataset.pkl')
    