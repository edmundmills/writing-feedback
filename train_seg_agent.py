import os

import numpy as np
import random
import transformers
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from core.d_elems import DElemTokenizer
from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.ner import NERTokenizer
from core.segmentation import make_agent
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
        args.seg.total_timesteps = 1024
        args.seg.n_envs = 2
    else:
        dataset = EssayDataset()

    train, val = dataset.split()

    ner_tokenizer = NERTokenizer(args.ner)
    d_elem_tokenizer = DElemTokenizer(args.kls)
    env = SegmentationEnv.make(args.seg.n_envs, train, ner_tokenizer, d_elem_tokenizer, args.env)

    with WandBRun(args, project_name='segmentation'):
        agent = make_agent(args, env)
        if args.wandb:
            callback = WandbCallback(verbose=args.seg.sb3_verbosity)
        else:
            callback = None
        agent.learn(total_timesteps=args.seg.total_timesteps,
                    callback=callback)
