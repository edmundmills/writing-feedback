import os

import numpy as np
import random
import transformers
import torch
from wandb.integration.sb3 import WandbCallback

from core.env import SegmentationEnv
from core.models.segmentation import SegmentationTokenizer, make_agent
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
        args.train_steps = 100
    else:
        dataset = EssayDataset()

    train, val = dataset.split()

    tokenizer = SegmentationTokenizer(args.ner)
    env = SegmentationEnv.make_vec(train, tokenizer, None, args)
    agent = make_agent(args, env)
 
    if args.wandb:
        callback = WandbCallback()
    else:
        callback = None


    with WandBRun(args):
        agent.learn(total_timesteps=args.train_steps, callback=callback)
