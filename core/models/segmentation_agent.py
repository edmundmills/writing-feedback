from typing import DefaultDict, Dict, List

import torch
import wandb

from core.env import SegmentationEnv
from core.model import Model
from core.models.essay_feedback import EssayModel
from utils.grading import to_predictions


def to_tokens(predictions, num_words):
    tokens = []
    for pred in predictions:
        tokens.append('START')
        tokens.extend(['CONT'] * (len(pred.word_idxs) - 1))
    while len(tokens) < num_words:
        tokens.append('MASK')
    return tokens


class SegmentationAgent(Model):
    def __init__(self, args):
        super().__init__()
        self.action_space_dim = args.action_space_dim # could possibly drop to 200
        # self.tokenizer = 
        # self.model = Longformer Classifier


    def encode(self,text:str):
        pass



    def act(self, state):
        encoded_pstrings = to_tokens(state.predictions, state.encoded_text.size()[0])
        with torch.no_grad():
            pass
            # plength = self.model(encoded_text, encoded_pstrings).item()
        # return plength

    def train(self, train_dataset, val_dataset, args) -> None:
        """
        Via Reinforcement Learning
        """
        # env = SegmentationEnv(train_dataset, self.tokenizer, self.argument_classifier)
        # use stablebaselines ppo
        for step in range(1, args.training_steps + 1):
            step_metrics = {}

    def eval(self, dataset, args, n_samples) -> Dict:
        metrics = DefaultDict(list) 
        for idx, essay in dataset:
            with torch.no_grad():
                pstrings = self(essay.text)
                logits = self.argument_classifier(essay.text, pstrings)
            predictions = to_predictions(pstrings, logits, essay.id)
            essay_metrics = essay.grade(predictions)
            for k, v in essay_metrics:
                metrics[k].append(v)

            if idx >= n_samples:
                metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
                break
        return metrics