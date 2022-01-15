import time
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import wandb

from core.constants import argument_names
from core.models.argument_encoder import ArgumentModel
from utils.grading import get_discourse_elements, get_labels
from utils.networks import Model, MLP, PositionalEncoder, Mode


class EssayDELemClassifier(nn.Module):
    def __init__(self, max_d_elems, num_attention_layers, nhead, linear_layers,
                 linear_layer_size, intermediate_layer_size, dropout):
        super().__init__()
        self.max_d_elems = max_d_elems
        self.num_encoder_layers = num_attention_layers
        self.nhead = nhead
        self.linear = MLP(n_inputs=769,
                           n_outputs=intermediate_layer_size,
                           n_layers=linear_layers,
                           layer_size=linear_layer_size,
                           dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermediate_layer_size,
                                                   nhead=nhead,
                                                   batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer, num_attention_layers)
        self.classifier = MLP(n_inputs=intermediate_layer_size,
                              n_outputs=len(argument_names),
                              n_layers=1,
                              layer_size=None)

    def forward(self, x):
        x = self.linear(x)
        x = self.attention(x)
        preds = self.classifier(x)
        return preds


class EssayModel(Model):
    def __init__(self, args, d_elem_encoder=None) -> None:
        super().__init__()
        self.max_d_elems = args.max_discourse_elements
        self.d_elem_encoder = d_elem_encoder or ArgumentModel()
        self.positional_encoder = PositionalEncoder(self.max_d_elems)
        self.essay_feedback = EssayDELemClassifier(
            max_d_elems=args.max_discourse_elements,
            num_attention_layers=args.num_attention_layers,
            nhead=args.nhead,
            linear_layers=args.num_linear_layers,
            linear_layer_size=args.linear_layer_size,
            intermediate_layer_size=args.intermediate_layer_size,
            dropout=args.dropout).to(self.device)

    def encode(self, sample):
        encoded_tensor = self.d_elem_encoder.encode(sample).cpu()
        encoded_tensor = self.positional_encoder(encoded_tensor)
        d_elem_lens = torch.LongTensor([len(d_elem.split()) for d_elem in sample]).unsqueeze(-1)
        encoded_tensor = torch.cat((d_elem_lens, encoded_tensor), dim=-1)
        padding_size = (max(0, self.max_d_elems - len(sample)), *encoded_tensor.size()[1:])
        padded_tensor = torch.cat((encoded_tensor[:self.max_d_elems, ...],
                                   -torch.ones(padding_size)), dim=0)
        return padded_tensor

    def inference(self, text:str, predictionstrings:List[str]):
        self.essay_feedback.eval()
        d_elems = get_discourse_elements(text, predictionstrings)
        encoded_text = self.encode(d_elems).to(self.device).unsqueeze(0)
        with torch.no_grad():
            preds = self.essay_feedback(encoded_text)
        preds = F.softmax(preds, dim=-1)
        self.essay_feedback.train()
        return preds

    # def train(self, train_dataset, val_dataset, args):
    #     with Mode(self.essay_feedback, mode='train'):
    #         optimizer = torch.optim.AdamW(self.essay_feedback.parameters(), lr=args.lr)

    #         running_loss = deque(maxlen=args.print_interval)
    #         timestamps = deque(maxlen=args.print_interval)
    #         step = 0

    #         for epoch in range(1, args.epochs + 1):
    #             print(f'Starting Epoch {epoch}')
    #             for essay in train_dataset:
    #                 pstrings = essay.random_pstrings()
    #                 labels = get_labels(pstrings, essay, num_d_elems=self.max_d_elems)
    #                 labels = torch.LongTensor(labels).to(self.device)
    #                 d_elems = get_discourse_elements(essay.text, pstrings)
    #                 with torch.no_grad():
    #                     encoded_text = self.encode(d_elems).to(self.device)

    #                 output = self.essay_feedback(encoded_text)
    #                 msk = (labels != -1)
    #                 output = output[msk]
    #                 labels = labels[msk]
    #                 loss = F.cross_entropy(output, labels)

    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.essay_feedback.parameters(), 1.0)
    #                 optimizer.step()

    #                 loss = loss.item()
    #                 running_loss.append(loss)
    #                 timestamps.append(time.time())
    #                 metrics = {'Train Loss': loss}
                    
    #                 if step % args.eval_interval == 0:
    #                     eval_metrics = self.eval(val_dataset, args, n_samples=(args.batches_per_eval * args.batch_size))
    #                     metrics.update(eval_metrics)
    #                     print(f'Step {step}:\t{eval_metrics}')

    #                 if args.wandb:
    #                     wandb.log(metrics)

    #                 if step % args.print_interval == 0:
    #                     print(f'Step {step}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
    #                         f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')
            
                    

    def train(self, train_dataset, val_dataset, args):
        dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                sampler=RandomSampler(train_dataset))
        
        with Mode(self.essay_feedback, mode='train'):
            optimizer = torch.optim.AdamW(self.essay_feedback.parameters(), lr=args.lr)

            running_loss = deque(maxlen=args.print_interval)
            timestamps = deque(maxlen=args.print_interval)
            step = 0

            for epoch in range(1, args.epochs + 1):
                print(f'Starting Epoch {epoch}')
                for (encoded_text, labels) in dataloader:
                    step += 1
                    encoded_text = encoded_text.to(self.device)
                    labels = labels.to(self.device)
                    output = self.essay_feedback(encoded_text)
                    labels = labels.squeeze(-1)
                    msk = (labels != -1)
                    output = output[msk]
                    labels = labels[msk]
                    loss = F.cross_entropy(output, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.essay_feedback.parameters(), 1.0)
                    optimizer.step()

                    loss = loss.item()
                    running_loss.append(loss)
                    timestamps.append(time.time())
                    metrics = {'Train Loss': loss}
                    
                    if step % args.eval_interval == 0:
                        eval_metrics = self.eval(val_dataset, args, n_samples=(args.batches_per_eval * args.batch_size))
                        metrics.update(eval_metrics)
                        print(f'Step {step}:\t{eval_metrics}')

                    if args.wandb:
                        wandb.log(metrics)

                    if step % args.print_interval == 0:
                        print(f'Step {step}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
                            f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')

        if args.save_model and args.wandb and not args.debug:
            self.save()

    def eval(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        with Mode(self.essay_feedback, mode='eval'):
            dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    sampler=RandomSampler(dataset))
            losses = []
            preds = []
            labels = []
            for step, (encoded_text, label) in enumerate(dataloader, start=1):
                label = label.to(self.device)
                encoded_text = encoded_text.to(self.device)
                with torch.no_grad():
                    output = self.essay_feedback(encoded_text)
                    label = label.squeeze(-1)
                    msk = (label != -1)
                    output = output[msk]
                    label = label[msk]
                    loss = F.cross_entropy(output, label)
                losses.append(loss.item())
                output = output.cpu().numpy()
                pred = np.argmax(output, axis=-1).flatten()
                label = label.cpu().numpy().flatten()
                preds.extend(pred)
                labels.extend(label)
                if step * args.batch_size >= n_samples:
                    break
            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(np.equal(preds, labels)) / len(preds)
        metrics = {'Eval Loss': avg_loss,
                   'Eval Accuracy': avg_acc}
        if args.wandb:
            confusion_matrix = wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=argument_names)
            metrics.update({'Confusion Matrix': confusion_matrix})
        return metrics
