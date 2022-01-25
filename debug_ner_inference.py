from pathlib import Path

import numpy as np
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.ner import NERTokenizer, NERModel
from core.essay import Prediction
from core.dataset import EssayDataset
from core.predicter import Predicter
from utils.config import parse_args, get_config
from utils.render import plot_ner_output, EssayRenderer


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    
    dataset = EssayDataset.load(args.ner_dataset_path)
    renderer = EssayRenderer()
    predicter = Predicter()
    dataset = EssayDataset.load(args.dataset_path)
    train, val = dataset.get_fold(0)
    tokenizer = NERTokenizer(args.ner)
    essay = val[0]
    val = tokenizer.make_ner_dataset([essay, essay])
    input_ids, attention_mask, labels, word_ids = val[0:1]

    ner_model = NERModel(args.ner)
    ner_model_name = f'{args.ner_model_name}_fold_0'
    model_dir = Path(f'./models/{ner_model.__class__.__name__}')
    model_file = next(model_dir.glob(f'*{ner_model_name}*')).stem
    ner_model.load(model_file)


    essay.ner_probs = ner_model.inference(input_ids,
                                            attention_mask,
                                            word_ids)


    segments = predicter.segment_ner_probs(essay.ner_probs)
    segment_lens = segments[:,:,-1].squeeze().tolist()
    preds, _grade = predicter.by_heuristics(essay, thresholds=False)
    renderer.render(essay, segment_lens=segment_lens, predictions=preds)
    