from typing import DefaultDict, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerModel
import wandb
from core.essay import Prediction

from core.model import Model
from utils.grading import to_predictions


class SegmentationModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.transformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.classifier = nn.Linear(768, 2)
        self.predictor = nn.Linear(1024*2, args.action_space_dim)

    def forward(self, tokenized_text, tokenized_predictions, attention_mask):
        attention_mask = tokenized_text == 1
        encoded = self.transformer(tokenized_text, attention_mask=attention_mask)
        logits = self.classifier(encoded.last_hidden_state)
        combined = torch.cat((logits[...,0], tokenized_predictions), dim=-1)
        pred = self.predictor(combined)
        pred = F.softmax(pred, dim=-1)
        return pred


class SegmentationAgent(Model):
    def __init__(self, args):
        super().__init__()
        self.action_space_dim = args.action_space_dim # could possibly drop to 200
        self.max_tokens = args.essay_max_tokens
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = SegmentationModel(args).to(self.device)

    def encode(self,text:str):
        tokenized = self.tokenizer.encode_plus(text,
                                             max_length=self.max_tokens,
                                             padding='max_length',
                                             truncation=True,
                                             return_tensors='pt',
                                             return_attention_mask=True)
        return tokenized

    def act(self, state):
        tokenized_preds = to_tokens(state.predictions, num_words=state.encoded_text.size(1))
        with torch.no_grad():
            preds = self.model(state.encoded_text.to(self.device),
                               tokenized_preds.to(self.device),
                               state.attention_mask.to(self.device)).cpu().squeeze().numpy()
        p_length = np.random.choice(self.action_space_dim, p=preds)
        pred = Prediction(state.word_idx, state.word_idx + p_length, -1, state.essay_id)
        return pred

    def train(self, env, args) -> None:
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