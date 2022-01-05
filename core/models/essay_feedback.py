import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import wandb

from core.constants import argument_names
from core.models.argument_encoder import ArgumentModel
from core.model import Model
from utils.grading import get_discourse_elements


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(d_model, max_seq_len)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[i + 1, pos] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        self.pe = pe

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        n_d_elems = x.size(0)
        with torch.no_grad():
            print(x.size(), self.pe.size(), self.pe[:n_d_elems,...].size())
            x = x + self.pe[:n_d_elems,...]
        return x

class EssayModel(Model):
    def __init__(self, args, d_elem_encoder=None) -> None:
        super().__init__()
        self.d_elem_encoder = d_elem_encoder or ArgumentModel()
        self.classifier = nn.Transformer(nhead=args.nhead,
                                         num_encoder_layers=args.num_encoder_layers,
                                         num_decoder_layers=args.num_decoder_layers,
                                         batch_first=True,
                                         device=self.device)
        self.token_len = args.n_discourse_elem_tokens
        self.positional_encoder = PositionalEncoder(self.token_len)

    def encode(self, sample):
        encoded_tensor = self.d_elem_encoder.encode(sample)
        encoded_tensor = self.positional_encoder(encoded_tensor)
        padding_size = (max(0, self.token_len - len(sample)), *encoded_tensor.size()[1:])
        padded_tensor = torch.cat((encoded_tensor[:self.token_len, ...], -torch.ones(padding_size)), dim=0)
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
