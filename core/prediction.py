import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import tqdm
import wandb

from utils.constants import ner_num_to_token, de_len_norm_factor, de_num_to_type
from utils.networks import Model, MLP, PositionalEncoder
from utils.render import plot_ner_output


class Predicter(Model):
    def __init__(self, pred_args):
        super().__init__()
        if pred_args.separate_seg_and_class_heads:
            self.two_headed = True
            self.num_outputs = len(de_num_to_type) + 2
        else:
            self.two_headed = False
            self.num_outputs = len(ner_num_to_token)
        self.seq_len = pred_args.num_ner_segments
        self.linear = MLP(n_inputs=pred_args.feature_dim,
                          n_outputs=pred_args.intermediate_layer_size,
                          n_layers=pred_args.num_linear_layers,
                          layer_size=pred_args.linear_layer_size).to(self.device)
        self.positional_encoder = PositionalEncoder(features=pred_args.intermediate_layer_size,
                                                    seq_len=self.seq_len,
                                                    device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=pred_args.intermediate_layer_size,
                                                   nhead=pred_args.n_head,
                                                   batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer,
                                               pred_args.num_attention_layers).to(self.device)
        self.head = MLP(n_inputs=pred_args.intermediate_layer_size,
                        n_outputs=self.num_outputs,
                        n_layers=1,
                        layer_size=None).to(self.device)
    
    def forward(self, features, plot_outputs=False):
        y = features
        # y = self.positional_encoder(y)
        # print(features.shape)
        if plot_outputs:
            plot_ner_output(y[0])
        y = self.linear(y)
        if plot_outputs:
            plot_ner_output(y[0])
        y = self.attention(y)
        if plot_outputs:
            plot_ner_output(y[0])
        y = self.head(y)
        if plot_outputs:
            plot_ner_output(y[0])
        return y

    def make_dataset(self, essay_dataset, args):
        print('Making NER Feature Dataset...')
        if args.predict.use_seg_t_features:
            print('Using Segment Transformer Features')
        features = []
        labels = []
        attention_masks = []
        for essay in tqdm.tqdm(essay_dataset):
            ner_features, _seg_lens, essay_labels = essay.segments
            if args.predict.use_seg_t_features:
                seg_t_features, attention_mask = essay.segment_tokens
                ner_features = seg_t_features.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            else:
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
        if self.two_headed:
            class_logits = output[...,:8]
            seg_logits = output[...,8:]
            cont_label = labels > 7
            class_labels = labels - (cont_label * 7)
            seg_labels = ((labels > 0) * (labels <= 7)).long()
            class_loss = F.cross_entropy(class_logits, class_labels)
            seg_loss = F.cross_entropy(seg_logits, seg_labels)
            loss = class_loss + 2*seg_loss
            metrics.update({
                'Train Loss': loss.item(),
                'Segmentation/Train Loss': seg_loss.item(),
                'Classification/Train Loss': class_loss.item()
            })
            if eval:
                seg_probs = F.softmax(seg_logits, dim=-1).cpu().numpy()
                seg_preds = np.argmax(seg_probs, axis=-1).flatten()
                seg_labels = seg_labels.cpu().numpy().flatten()
                class_probs = F.softmax(class_logits, dim=-1).cpu().numpy()
                class_preds = np.argmax(class_probs, axis=-1).flatten()
                class_labels = class_labels.cpu().numpy().flatten()
                metrics.update({
                    'Eval Loss': loss.item(),
                    'Seg Probs': seg_probs,
                    'Seg Preds': seg_preds,
                    'Seg Labels': seg_labels,
                    'Class Probs': class_probs,
                    'Class Preds': class_preds,
                    'Class Labels': class_labels,
                    'Segmentation/Eval Loss': seg_loss.item(),
                    'Classification/Eval Loss': class_loss.item()
                })
        else:
            loss = F.cross_entropy(output, labels)
            metrics.update({
                'Train Loss': loss.item()
            })
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
        def flatten_list(nested_list):
            return np.array([item for sublist in nested_list for item in sublist])

        def calculate_metrics(loss, preds, labels):
            avg_loss = sum(loss) / len(loss)
            correct = np.equal(preds, labels)
            avg_acc = sum(correct) / len(correct)            
            seg_confusion_matrix = wandb.plot.confusion_matrix(
                y_true=labels,
                preds=preds,
                class_names=ner_num_to_token)
            metrics = {'Eval Loss': avg_loss,
                       'Eval Accuracy': avg_acc,
                       'Confusion Matrix': seg_confusion_matrix}
            return metrics

        eval_metrics = {}
        if self.two_headed:
            seg_preds = flatten_list(metrics['Seg Preds'])
            seg_labels = flatten_list(metrics['Seg Labels'])
            class_preds = flatten_list(metrics['Class Preds'])
            class_labels = flatten_list(metrics['Class Labels'])
            seg_metrics = calculate_metrics(metrics['Segmentation/Eval Loss'],
                                            seg_preds, seg_labels)
            class_metrics = calculate_metrics(metrics['Classification/Eval Loss'],
                                            class_preds, class_labels)
            eval_metrics.update({f'Segmentation/{k}': v for k, v in seg_metrics.items()})
            eval_metrics.update({f'Classification/{k}': v for k, v in class_metrics.items()})
        else:
            preds = flatten_list(metrics['Preds'])
            labels = flatten_list(metrics['Labels'])
            metrics = calculate_metrics(metrics['Eval Loss'], preds, labels)
            eval_metrics.update({f'NER/{k}': v for k, v in metrics.items()})
        return eval_metrics