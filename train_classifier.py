import numpy as np
import pickle
import random
import torch
from transformers import data
import wandb

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.predicter import Predicter
from core.dataset import EssayDataset
from utils.config import parse_args, get_config, WandBRun


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    ner_dataset = EssayDataset.load(args.ner_dataset_path)
    if not args.predict.load_ner_features:
        if args.debug:
            dataset = EssayDataset(n_essays=20)
            dataset.ner_probs = ner_dataset.ner_probs
            dataset.make_folds(2)
            args.predict.print_interval = 10
            args.predict.eval_interval = 10
            args.predict.eval_samples = 5
            args.predict.epochs = 10
        else:
            dataset = ner_dataset
    else:
        dataset = ner_dataset

    first_run_name = None

    for fold in dataset.folds:
        predicter = Predicter(args.predict)
        print(f'Starting training on fold {fold}')

        if not args.predict.load_ner_features:
            train, val = dataset.get_fold(fold)
            train = predicter.make_ner_feature_dataset(train)
            val = predicter.make_ner_feature_dataset(val)
            if args.predict.save_ner_features and not args.debug:
                with open(f'data/ner_features_fold_{fold}_train', 'wb') as filename:
                    pickle.dump(train, filename)
                with open(f'data/ner_features_fold_{fold}_val', 'wb') as filename:
                    pickle.dump(val, filename)
        else:
            with open(f'data/ner_features_fold_{fold}_train', 'rb') as filename:
                train = pickle.load(filename)
            with open(f'data/ner_features_fold_{fold}_val', 'rb') as filename:
                val = pickle.load(filename)
        print(f'Training dataset size: {len(train)}')
        print(f'Validation dataset size: {len(val)}')

        with WandBRun(args, project_name='classifier'):
            run_name = wandb.run.name if args.wandb else 'test'
            first_run_name = first_run_name or run_name
            classifier = predicter.classifier
            classifier.learn(train, val, args)
            if args.ner.save_model:
                model_name = f'{first_run_name}_fold_{fold}_{wandb.run.name}' if args.wandb else 'test'
                classifier.save(model_name)
