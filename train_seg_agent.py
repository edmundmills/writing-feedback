import os

import numpy as np
import random
import transformers
import torch
from wandb.integration.sb3 import WandbCallback

from core.d_elems import DElemTokenizer
from core.dataset import EssayDataset
from core.env import DividerEnv, SequencewiseEnv
from core.ner import NERTokenizer
from core.segmentation import make_agent
from utils.config import parse_args, get_config, WandBRun


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

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

    ner_tokenizer = NERTokenizer(args.ner)
    d_elem_tokenizer = DElemTokenizer(args.kls)
    env = DividerEnv.make_vec(args.seg.envs, train, ner_tokenizer, d_elem_tokenizer, args.env)
 
    with WandBRun(args):
        agent = make_agent(args, env)
        if args.wandb:
            callback = WandbCallback(
                gradient_save_freq=100,
                verbose=2)
        else:
            callback = None
        agent.learn(total_timesteps=args.seg.train_steps, callback=callback)
