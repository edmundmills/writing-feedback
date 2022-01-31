import os
from turtle import position

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
from core.env import SegmentationEnv
from utils.config import parse_args, get_config
from utils.render import plot_ner_output


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)
    
    dataset = EssayDataset.load(args.ner_dataset_path)

    env = SegmentationEnv.make(1, dataset, args)

    state = env.reset()
    print(env.essay_id)
    done = False
    while not done:
        plot_ner_output(state)
        action = int(input('Action: '))
        # print(env.actions[action])
        state, reward, done, info = env.step(action)
        print(f'Reward: {reward:.2f}')
    grade = info['Score']
    print(f'f-Score: {grade}')
    # for pred in env.essay.correct_predictions:
    #     print(pred.start, pred.stop, pred.label)
    # for pred in env.predictions:
    #     print(pred.start, pred.stop, pred.label)