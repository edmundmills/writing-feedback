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
from core.prediction import Predicter
from utils.config import parse_args, get_config, WandBRun
from utils.constants import de_len_norm_factor
from utils.render import plot_ner_output

if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)


    ner_dataset = EssayDataset.load(args.segmented_dataset_path)
    if args.debug:
        dataset = EssayDataset(n_essays=2)
        dataset.copy_essays(ner_dataset)
        args.rl.total_timesteps = 4096*20
        args.rl.n_envs = min(1, args.rl.n_envs)
    else:
        dataset = ner_dataset

    train, val = dataset.split()

    # env = SegmentationEnv.make(args.rl.n_envs, train, args)

    predicter = Predicter(args.predict)
    predicter.load(args.rl.saved_model_name)
    essay = dataset[0]
    segmented_ner_probs, segment_lens, labels = essay.segments
    segmented_ner_probs[...,-1] /= de_len_norm_factor
    plot_ner_output(segmented_ner_probs)
    with torch.no_grad():
        output = predicter(segmented_ner_probs.to(predicter.device)).cpu()
    probs = torch.softmax(output, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    print(labels)
    print(pred)
    plot_ner_output(probs)