import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb

from core.dataset import EssayDataset
from core.ner import NERModel, NERTokenizer
from utils.config import parse_args, get_config
from utils.constants import ner_probs_path


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
    else:
        dataset = EssayDataset()


    ner_tokenizer = NERTokenizer(args.ner)
    ner_model = NERModel(args.ner)
    ner_model.load(args.ner.ner_model_name)
    dataset.ner_probs = ner_model.infer_for_dataset(dataset, ner_tokenizer)
    print(f'Saving NER Probs to {ner_probs_path}')
    with open(ner_probs_path, 'wb') as saved_file:
        pickle.dump(dataset.ner_probs, saved_file)
    print('NER probs saved')

