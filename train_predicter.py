import numpy as np
import random
import torch
import wandb

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.prediction import Predicter
from core.segmenter import Segmenter
from core.segment_transformer import SegmentTransformer
from utils.config import parse_args, get_config, WandBRun


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    if not args.predict.load_ner_features:
        ner_dataset = EssayDataset.load(args.baseline_ner_dataset_path)
        segmenter = Segmenter(args.seg)
        ner_dataset = segmenter.segment_essay_dataset(ner_dataset, print_avg_grade=True)
        if args.predict.save_ner_features and not args.debug:
            ner_dataset.save(args.segmented_dataset_path)
    elif args.predict.use_combined_features:
        ner_dataset = EssayDataset.load(args.combined_features_dataset_path)
    elif args.predict.use_seg_t_features:
        ner_dataset = EssayDataset.load(args.tokenized_dataset_path)
    else:
        ner_dataset = EssayDataset.load(args.segmented_dataset_path)

    if args.debug:
        dataset = EssayDataset(n_essays=20)
        dataset.copy_essays(ner_dataset)
        dataset.make_folds(2)
        args.predict.print_interval = 10
        args.predict.eval_interval = 10
        args.predict.eval_samples = 5
        args.predict.epochs = 20
    else:
        dataset = ner_dataset


    first_run_name = None

    for fold in dataset.folds:
        with WandBRun(args, project_name='classifier'):
            print(f'Starting training on fold {fold}')
            train, val = dataset.get_fold(fold)
            predicter = Predicter(args.predict)
            train_tensor = predicter.make_dataset(train, args)
            val_tensor = predicter.make_dataset(val, args)
            print(f'Training dataset size: {len(train_tensor)}')
            print(f'Validation dataset size: {len(val_tensor)}')

            run_name = wandb.run.name if args.wandb else 'test'
            first_run_name = first_run_name or run_name
            predicter.learn(train_tensor, val_tensor, args.predict)
            if args.predict.save_output_features:
                val = predicter.infer_on_dataset(val, args)
                dataset.copy_essays(val)

            if args.ner.save_model:
                model_name = f'{first_run_name}_fold_{fold}_{wandb.run.name}' if args.wandb else 'test'
                predicter.save(model_name)

    if args.predict.save_output_features:
        dataset.save(args.combined_features_dataset_path)