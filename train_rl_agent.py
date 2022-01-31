import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb
from wandb.integration.sb3 import WandbCallback


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.agent import make_agent
from utils.config import parse_args, get_config, WandBRun



if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    if args.wandb:
        wandb.tensorboard.patch(root_logdir="log")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()

    ner_dataset = EssayDataset.load(args.segmented_dataset_path)
    if args.debug:
        dataset = EssayDataset(n_essays=2)
        dataset.copy_essays(ner_dataset)
        args.rl.total_timesteps = 4096*20
        args.rl.n_envs = min(1, args.rl.n_envs)
    else:
        dataset = ner_dataset

    train, val = dataset.split()

    env = SegmentationEnv.make(args.rl.n_envs, train, args)

    with WandBRun(args, project_name='segmentation'):
        agent = make_agent(args, env)
        if args.wandb:
            callback = WandbCallback(verbose=args.rl.sb3_verbosity)
        else:
            callback = None
        agent.learn(total_timesteps=args.rl.total_timesteps,
                    callback=callback)
