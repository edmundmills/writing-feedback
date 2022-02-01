import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import tqdm
import wandb

from utils.constants import ner_num_to_token, de_len_norm_factor
from utils.networks import Model, MLP, PositionalEncoder
from utils.render import plot_ner_output


class Predicter(Model):
    def __init__(self, pred_args):
        super().__init__()
        self.num_outputs = len(ner_num_to_token)
        self.seq_len = pred_args.num_ner_segments
        self.positional_encoder = PositionalEncoder(features=pred_args.feature_dim,
                                                    seq_len=self.seq_len,
                                                    device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=pred_args.feature_dim,
                                                   nhead=pred_args.n_head,
                                                   batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer,
                                               pred_args.num_attention_layers).to(self.device)
        self.head = MLP(n_inputs=pred_args.feature_dim,
                          n_outputs=self.num_outputs,
                          n_layers=pred_args.num_linear_layers,
                          layer_size=pred_args.linear_layer_size).to(self.device)
    
    def forward(self, features, plot_outputs=False):
        # y = self.positional_encoder(features)
        # print(features.shape)
        if plot_outputs:
            plot_ner_output(features[0])
        y = self.attention(features)
        if plot_outputs:
            plot_ner_output(y[0])
        y = self.head(y)
        if plot_outputs:
            plot_ner_output(y[0])
        return y

    def make_ner_feature_dataset(self, essay_dataset):
        print('Making NER Feature Dataset...')
        features = []
        labels = []
        attention_masks = []
        for essay in tqdm.tqdm(essay_dataset):
            ner_features, _seg_lens, essay_labels = essay.segments
            ner_features[...,-1] /= de_len_norm_factor
            attention_mask = (ner_features[...,0] != -1)
            attention_masks.append(attention_mask.bool())
            features.append(ner_features)
            labels.append(essay_labels)
        features = torch.cat(features, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0).unsqueeze(-1)
        dataset = TensorDataset(features, attention_masks, labels)
        print(f'Dataset created with {len(dataset)} samples')
        return dataset

    def loss(self, sample, eval=False):
        metrics = {}
        ner_features, attention_mask, labels = sample
        ner_features = ner_features.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        output = self(ner_features)
        output = output[attention_mask]
        labels = labels[attention_mask].squeeze()
        loss = F.cross_entropy(output, labels)
        if eval:
            probs = F.softmax(output, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1).flatten()
            labels = labels.cpu().numpy().flatten()
            metrics.update({
                'Eval Loss': loss.item(),
                'Probs': probs,
                'Preds': preds,
                'Labels': labels,
            })
        return loss, metrics

    def process_eval_metrics(self, metrics):
        avg_loss = sum(metrics['Eval Loss']) / len(metrics['Eval Loss'])
        preds = np.array([pred for sublist in metrics['Preds']
                          for pred in sublist])
        labels = np.array([label for sublist in metrics['Labels']
                           for label in sublist])
        correct = np.equal(preds, labels)
        avg_acc = sum(correct) / len(correct)            
        eval_metrics = {'Eval Loss': avg_loss,
                        'Eval Accuracy': avg_acc}
        seg_confusion_matrix = wandb.plot.confusion_matrix(
            y_true=labels,
            preds=preds,
            class_names=ner_num_to_token)
        eval_metrics.update({'Confusion Matrix': seg_confusion_matrix})
        return eval_metrics