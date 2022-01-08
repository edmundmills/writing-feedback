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
from utils.networks import MLP, PositionalEncoder


        super().__init__()

class EssayModel(Model):
    def __init__(self, args, d_elem_encoder=None) -> None:
        super().__init__()
        self.d_elem_encoder = d_elem_encoder or ArgumentModel()
        encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                   nhead=args.nhead,
                                                   batch_first=True,
                                                   device=self.device)
        self.encoder = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers)
        self.decoder = nn.Conv2d(in_channels=1, out_channels=len(argument_names), kernel_size=(1, 768), device=self.device)
        self.max_d_elems = args.max_discourse_elements
        self.positional_encoder = PositionalEncoder(self.max_d_elems)

    def encode(self, sample):
        encoded_tensor = self.d_elem_encoder.encode(sample).cpu()
        encoded_tensor = self.positional_encoder(encoded_tensor)
        padding_size = (max(0, self.max_d_elems - len(sample)), *encoded_tensor.size()[1:])
        padded_tensor = torch.cat((encoded_tensor[:self.max_d_elems, ...],
                                   -torch.ones(padding_size)), dim=0)
        return padded_tensor

    def inference(self, text:str, predictionstrings:List[str]):
        d_elems = get_discourse_elements(text, predictionstrings)
        encoded_text = self.encode(d_elems).to(self.device).unsqueeze(0)
        with torch.no_grad():
            preds = self(encoded_text)
        return preds

    def forward(self, src):
        encoded = self.encoder(src).unsqueeze(-3)
        preds = self.decoder(encoded)
        preds = torch.transpose(preds, 1, 2).squeeze(-1)
        preds = F.softmax(preds, dim=-1)
        return preds

    def train(self, train_dataset, val_dataset, args):
        dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                sampler=RandomSampler(train_dataset))
        
        self.encoder.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            print(f'Starting Epoch {epoch}')
            for encoded_text, labels in dataloader:
                encoded_text = encoded_text.to(self.device)
                labels = labels.to(self.device)
                output = self(src=encoded_text)
                output = output.reshape(-1, 8)
                labels = labels.reshape(-1)
                msk = (labels != -1)
                output = output[msk]
                labels = labels[msk]
                loss = F.cross_entropy(output, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()

                metrics = {'Train Loss': loss.item()}
                print(metrics)
                if args.wandb:
                    wandb.log(metrics)


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
