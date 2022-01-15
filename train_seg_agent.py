import os

import numpy as np
import random
import transformers
import torch
from wandb.integration.sb3 import WandbCallback

from core.env import SequencewiseEnv
from core.ner import NERTokenizer
from core.segmentation import make_agent
from core.dataset import EssayDataset

from utils.config import parse_args, get_config, WandBRun

if __name__ == '__main__':
    args = parse_args()
    args = get_config('segmentation', args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.debug:
        dataset = EssayDataset(n_essays=200)
        args.seg.train_steps = 100
    else:
        dataset = EssayDataset()

    train, val = dataset.split()

    tokenizer = NERTokenizer(args.ner)
    env = SequencewiseEnv.make_vec(train, tokenizer, None, args)
 
    with WandBRun(args):
        agent = make_agent(args, env)
        if args.wandb:
            callback = WandbCallback(
                gradient_save_freq=100,
                verbose=2)
        else:
            callback = None
        agent.learn(total_timesteps=args.train_steps, callback=callback)
