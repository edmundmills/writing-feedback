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

from core.bc import BCAgent, make_bc_dataset
from core.dataset import EssayDataset
from core.env import SegmentationEnv
from utils.config import parse_args, get_config, WandBRun
from utils.constants import ner_probs_path


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir="log")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    if args.debug:
        dataset = EssayDataset(n_essays=3)
    else:
        dataset = EssayDataset()


    with open(ner_probs_path, 'rb') as saved_file:
        dataset.ner_probs = pickle.load(saved_file)
    print(f'NER Probs Loaded from {ner_probs_path}')
    train, val = dataset.split()

    env = SegmentationEnv.make(1, train, args.env)
    bc_dataset = make_bc_dataset(train, env)

    with WandBRun(args, project_name='segmentation'):
        agent = BCAgent(args)
        agent.learn(bc_dataset, env)