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
        self.classifier = nn.Transformer(nhead=args.nhead,
                                         num_encoder_layers=args.num_encoder_layers,
                                         num_decoder_layers=args.num_decoder_layers)
        self.token_len = args.n_discourse_elem_tokens

    def encode(self, sample):
        token_tensor = self.encoder.encode(sample, self.token_len)
        padding_size = (max(0, self.token_len - len(sample)), *token_tensor.size()[1:])
        padded_tensor = torch.cat((token_tensor[:self.token_len, ...], torch.zeros(padding_size)), dim=0)
        return padded_tensor

    def forward(self, text:str, predictionstrings:List[str]):
        pass

    def train(self, train_dataset, val_dataset, args):
        pass

    def eval(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        self.encoder.eval()
        # self.classifier.eval()
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(dataset))
        losses = []
        preds = []
        labels = []
        for step, (sample, label) in enumerate(dataloader):
            gpu_label = label.to(self.device)
            with torch.no_grad():
                encodings = self.encode(sample)
                logits = self.classifier(encodings)
                loss = F.cross_entropy(logits, gpu_label)
            losses.append(loss.item())
            logits = logits.cpu().numpy()
            pred = np.argmax(logits, axis=1).squeeze()
            label = label.numpy().squeeze()
            preds.extend(pred)
            labels.extend(label)
            if (step + 1) * args.batch_size >= n_samples:
                break
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(np.equal(pred, label)) / len(pred)
        self.encoder.train()
        confusion_matrix = wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=argument_names)
        return {'Type Classifier/Eval Loss': avg_loss,
                'Type Classifier/Eval Accuracy': avg_acc,
                'Type Classifier/Confusion Matrix': confusion_matrix}
