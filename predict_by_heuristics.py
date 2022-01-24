import numpy as np
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.predicter import Predicter
from utils.config import parse_args, get_config
from utils.render import plot_ner_output


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    dataset = EssayDataset.load(args.ner_dataset_path)

    predicter = Predicter()

    scores = []
    count = 100
    for essay in dataset:
        _preds, metrics = predicter.by_heuristics(essay)
        score = metrics['f_score']
        scores.append(score)
        if count and len(scores > count): break
    avg_score = sum(scores) / len(scores)
    print(avg_score)