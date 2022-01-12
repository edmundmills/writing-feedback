import os

import numpy as np
import random
import transformers
import torch

from core.env import SegmentationEnv
from core.models.segmentation_agent import SegmentationAgent
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
        args.print_interval = 10
        args.eval_interval = 50
        args.batches_per_eval = 4
        args.epochs = max(args.epochs, 20)
    else:
        dataset = EssayDataset()

    train, val = dataset.split()

    agent = SegmentationAgent(args)
    env = SegmentationEnv(train, agent, None)

    with WandBRun(args):
        state = env.reset()
        act = agent.act(state)