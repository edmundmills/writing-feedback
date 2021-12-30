from typing import DefaultDict, Dict, List

import torch
import wandb

from core.env import SegmentationEnv
from core.model import Model
from core.models.essay_feedback import EssayModel
from utils.grading import pstrings_to_tokens, to_predictions


class SegmentationAgent(Model):
    def __init__(self):
        super().__init__()
        self.n_outputs = 256 # could possibly drop to 200
        # self.tokenizer = 
        # self.model = Longformer Classifier
        self.argument_classifier = EssayModel()
        self.argument_classifier.eval()

    def __call__(self, text:str) -> List[str]:
        pass

    def act(self, encoded_text, pstrings):
        encoded_pstrings = pstrings_to_tokens(pstrings, encoded_text.size()[0])
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