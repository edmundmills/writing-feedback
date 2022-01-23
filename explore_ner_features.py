import os

import matplotlib.pyplot as plt
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
from core.predicter import Predicter
from utils.config import parse_args, get_config
from utils.render import plot_ner_output  


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir="log")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    dataset = EssayDataset(10)
    ner_probs_path = 'data/ner_probs.pkl'
    with open(ner_probs_path, 'rb') as saved_file:
        dataset.ner_probs = pickle.load(saved_file)
    print(f'NER Probs Loaded from {ner_probs_path}')

    essay = dataset[0]
    print(essay.essay_id, len(essay.words))
    for pred in essay.correct_predictions:
        print(pred)
    ner_probs = dataset.ner_probs[essay.essay_id]
    segments = Predicter().segment_ner_probs(ner_probs)
    plot_ner_output(ner_probs)
    plot_ner_output(segments)