import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.rl import make_agent
from utils.config import parse_args, get_config, WandBRun



if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir="log")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.debug:
        dataset = EssayDataset(n_essays=3)
        args.rl.total_timesteps = 4096
        args.rl.n_envs = min(2, args.rl.n_envs)
    else:
        dataset = EssayDataset()


    ner_probs_path = 'data/ner_probs.pkl'
    with open(ner_probs_path, 'rb') as saved_file:
        dataset.ner_probs = pickle.load(saved_file)
    print(f'NER Probs Loaded from {ner_probs_path}')

    train, val = dataset.split()

    env = SegmentationEnv.make(args.rl.n_envs, train, args.env)

    with WandBRun(args, project_name='segmentation'):
        agent = make_agent(args, env)
        if args.wandb:
            callback = WandbCallback(verbose=args.rl.sb3_verbosity)
        else:
            callback = None
        agent.learn(total_timesteps=args.rl.total_timesteps,
                    callback=callback)
