from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import wandb

from core.models.argument_encoder import ArgumentModel
from core.model import Model

class EssayModel(Model):
    def __init__(self, args) -> None:
        super().__init__()
        self.encoder = ArgumentModel()
        # self.classifier = transformer()

    def encode(self, sample):
        return self.encoder.encode(sample)

    def forward(self, text:str, predictionstrings:List[str]):
        pass

    def train(self, train_dataset, val_dataset, args):
        pass

