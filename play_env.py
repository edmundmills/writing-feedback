import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from core.d_elems import DElemTokenizer
from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.ner import NERModel, NERTokenizer
from core.segmentation import make_agent
from utils.config import parse_args, get_config, WandBRun
from utils.render import plot_ner_output


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

    dataset = EssayDataset(2)


    ner_probs_path = 'data/ner_probs.pkl'
    with open(ner_probs_path, 'rb') as saved_file:
        dataset.ner_probs = pickle.load(saved_file)
    print(f'NER Probs Loaded from {ner_probs_path}')

    env = SegmentationEnv.make(1, dataset, args.env)

    state = env.reset()
    print(env.essay_id)
    done = False
    while not done:
        plot_ner_output(state)
        action = int(input('Action: '))
        print(env.actions[action])
        state, reward, done, info = env.step(action)
        print(f'Reward: {reward:.2f}')
    grade = env.current_state_value()
    print(f'f-Score: {grade}')
    for pred in env.essay.correct_predictions:
        print(pred.start, pred.stop, pred.label)
    for pred in env.predictions:
        print(pred.start, pred.stop, pred.label)