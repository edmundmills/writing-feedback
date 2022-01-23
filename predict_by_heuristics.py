import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.essay import Prediction
from core.predicter import Predicter
from utils.config import parse_args, get_config
from utils.postprocessing import link_evidence, proba_thresh, min_thresh
from utils.render import plot_ner_output


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir="log")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    dataset = EssayDataset(100)


    ner_probs_path = 'data/ner_probs.pkl'
    with open(ner_probs_path, 'rb') as saved_file:
        dataset.ner_probs = pickle.load(saved_file)
    print(f'NER Probs Loaded from {ner_probs_path}')

    predicter = Predicter()

    scores = []
    for essay in dataset:
        _preds, metrics = predicter.by_heuristics(essay)
        score = metrics['f_score']
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    print(avg_score)