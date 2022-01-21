import os

import numpy as np
import pickle
import random
import transformers
import torch
import wandb

from core.dataset import EssayDataset
from core.env import SegmentationEnv
from core.essay import Prediction
from utils.config import parse_args, get_config
from utils.postprocessing import link_evidence, proba_thresh, min_thresh
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

    dataset = EssayDataset(100)


    ner_probs_path = 'data/ner_probs.pkl'
    with open(ner_probs_path, 'rb') as saved_file:
        dataset.ner_probs = pickle.load(saved_file)
    print(f'NER Probs Loaded from {ner_probs_path}')

    start_thresh = 0.8
    scores = []
    for essay in dataset:
        ner_probs = dataset.ner_probs[essay.essay_id]
        start_preds = ner_probs[:,:,0].squeeze() > start_thresh
        class_probs = ner_probs[:,:,1:]
        class_preds = np.argmax(class_probs, axis=-1).squeeze()
        predictions = []
        for idx, (start_pred, class_pred) in enumerate(zip(start_preds, class_preds)):
            if idx == 0:
                cur_pred_start = 0
                cur_pred_class = class_preds[0]
                continue
            if class_pred == cur_pred_class and not start_pred:
                continue
            pred = Prediction(cur_pred_start, idx, int(cur_pred_class), essay.essay_id)
            pred_weights = class_probs[0, pred.start:(pred.stop + 1), pred.label]
            class_confidence = sum(pred_weights.squeeze().tolist()) / len(pred_weights.squeeze().tolist())
            if class_confidence > proba_thresh[pred.argument_name] and len(pred) > min_thresh[pred.argument_name]:
                predictions.append(pred.formatted())
            cur_pred_class = class_pred
            cur_pred_start = idx
        score = essay.grade(predictions)['f_score']
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    print(avg_score)