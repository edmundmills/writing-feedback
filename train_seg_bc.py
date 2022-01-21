import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb

from core.bc import BCAgent
from core.d_elems import DElemTokenizer
from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.ner import NERModel, NERTokenizer
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
        args.seg.n_envs = min(2, args.seg.n_envs)
    else:
        dataset = EssayDataset()


    ner_probs_path = 'data/ner_probs.pkl'
    if args.seg.load_ner_probs:
        with open(ner_probs_path, 'rb') as saved_file:
            dataset.ner_probs = pickle.load(saved_file)
        print(f'NER Probs Loaded from {ner_probs_path}')
    else:
        ner_tokenizer = NERTokenizer(args.ner)
        d_elem_tokenizer = DElemTokenizer(args.kls)
        ner_model = NERModel(args.ner)
        ner_model.load(args.seg.ner_model_name)
        dataset.get_ner_probs(ner_tokenizer, ner_model)
        del ner_tokenizer
        del ner_model
        if args.seg.save_ner_probs:
            print(f'Saving NER Probs to {ner_probs_path}')
            with open(ner_probs_path, 'wb') as saved_file:
                pickle.dump(dataset.ner_probs, saved_file)
            print('NER probs saved')

    train, val = dataset.split()

    env = SegmentationEnv.make(args.seg.n_envs, train, args.env)

    bc_dataset = train.make_bc_dataset(env)

    with WandBRun(args, project_name='segmentation'):
        agent = BCAgent(args)
        agent.learn(bc_dataset, env)