from collections import deque, namedtuple
import time
from typing import Union, List

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel
import tqdm
import numpy as np
import wandb

from core.dataset import SegmentTokens, Segments, EssayDataset
from core.essay import Prediction
from utils.constants import ner_num_to_token, de_num_to_type
from utils.networks import Model, MLP, PositionalEncoder, Mode




class SegmentTokenizer(SentenceTransformer):
    def __init__(self, pred_args):
        print('Loading Segment Tokenizer...')
        super().__init__('sentence-transformers/all-distilroberta-v1')
        self.max_d_elems = pred_args.num_ner_segments
        self.max_seq_len = 768
        print('Segment Tokenizer Loaded')

    def encode(self, segments):
        num_segments = len(segments)
        encoded = super().encode(segments, convert_to_numpy=False, convert_to_tensor=True).cpu()
        attention_mask = torch.ones_like(encoded)
        encoded = torch.cat((encoded[:self.max_d_elems,:],
                                    torch.ones(self.max_d_elems - num_segments, self.max_seq_len)))
        attention_mask = torch.cat((attention_mask[:self.max_d_elems,:],
                                    torch.zeros(self.max_d_elems - num_segments, self.max_seq_len)))
        return encoded, attention_mask[...,0].bool()

    def tokenize_dataset(self, essay_dataset):
        print('Tokenizing Dataset Segments...')
        for essay in tqdm.tqdm(essay_dataset):
            _essay_ner_features, seg_lens, essay_labels = essay.segments
            text = essay.text_from_segments(seg_lens, join=True)
            encoded, attention_mask = self.encode(text)
            segment_tokens = SegmentTokens(encoded, attention_mask)
            essay_dataset.essays[essay.essay_id]['segment_tokens'] = segment_tokens
        print(f'Dataset Tokenized')
        return essay_dataset


class SegmentClassifier(Model):
    def __init__(self, pred_args) -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base").to(self.device)
        self.use_ner_probs = pred_args.use_ner_probs
        self.use_seg_lens = pred_args.use_seg_lens
        self.seq_len = pred_args.num_ner_segments
        self.num_outputs = 10
        mlp_inputs = 768
        if self.use_ner_probs:
            mlp_inputs += 15
        if self.use_seg_lens:
            mlp_inputs += 1
        self.head = MLP(n_inputs=mlp_inputs,
                        n_outputs=self.num_outputs,
                        n_layers=pred_args.num_linear_layers,
                        dropout=pred_args.dropout,
                        layer_size=pred_args.linear_layer_size).to(self.device)
        self.positional_encoder = PositionalEncoder(features=768,
                                                    seq_len=self.seq_len,
                                                    device=self.device)


    def forward(self, input_ids, attention_mask, ner_probs, seg_lens):
        input_ids = self.positional_encoder(input_ids)
        output = self.model(inputs_embeds=input_ids, attention_mask=attention_mask)
        y = output['last_hidden_state']
        features = [y]
        if self.use_ner_probs:
            features.append(ner_probs)
        if self.use_seg_lens:
            features.append(seg_lens)
        y = torch.cat(features, dim=-1)
        y = self.head(y)
        class_logits = y[...,:8]
        seg_logits = y[...,8:]
        return class_logits, seg_logits

    def predict(self, essay):
        ner_features, seg_lens, essay_labels = essay.segments
        input_ids, attention_mask = essay.segment_tokens
        ner_probs = ner_features[...,:-1]
        seg_lens = ner_features[...,-1:] / 200
        with Mode(self, 'eval'):
            with torch.no_grad():
                print(input_ids.shape, attention_mask.shape, ner_probs.shape, seg_lens.shape)
                class_logits, seg_logits = self(input_ids.to(self.device),
                                                attention_mask.to(self.device),
                                                ner_probs.to(self.device),
                                                seg_lens.to(self.device))
                class_probs = F.softmax(class_logits, dim=-1).cpu()
                class_preds = torch.argmax(class_probs, dim=-1)
                seg_probs = F.softmax(seg_logits, dim=-1).cpu()
                seg_preds = torch.argmax(seg_probs, dim=-1)

        preds = []
        cur_start = 0
        word_idx = 0
        for idx, (seg_len, class_pred, class_prob, seg_pred) in enumerate(
                zip(seg_lens, class_preds, class_probs, seg_preds)):
            word_idx += seg_len
            if idx == 0:
                cur_class = class_pred.item()
                continue
            if seg_pred.item() or class_pred.item() != cur_class:
                preds.append(Prediction(cur_start, word_idx - 1, cur_class, essay.essay_id))
                cur_start = word_idx
                cur_class = class_pred.item()
            preds.append(Prediction(cur_start, word_idx - 1, cur_class, essay.essay_id))
        grade = essay.grade[preds]
        return preds, grade            


    def loss(self, sample, eval=False):
        metrics = {}
        input_ids, ner_features, attention_mask, labels = sample
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        ner_features = ner_features.to(self.device)
        ner_probs = ner_features[...,:-1]
        seg_lens = ner_features[...,-1:] / 200
        class_logits, seg_logits = self(input_ids, attention_mask, ner_probs, seg_lens)
        class_logits = class_logits[attention_mask]
        seg_logits = seg_logits[attention_mask]
        labels = labels[attention_mask].squeeze()
        cont_label = labels > 7
        class_labels = labels - (cont_label * 7)
        seg_labels = ((labels > 0) * (labels <= 7)).long()
        class_loss = F.cross_entropy(class_logits, class_labels)
        seg_loss = F.cross_entropy(seg_logits, seg_labels)
        loss = class_loss + 2*seg_loss
        if eval:
            class_probs = F.softmax(class_logits, dim=-1).cpu().numpy()
            class_preds = np.argmax(class_probs, axis=-1).flatten()
            class_labels = class_labels.cpu().numpy().flatten()
            seg_probs = F.softmax(seg_logits, dim=-1).cpu().numpy()
            seg_preds = np.argmax(seg_probs, axis=-1).flatten()
            seg_labels = seg_labels.cpu().numpy().flatten()
            metrics.update({
                'Eval/Loss': loss.item(),
                'Eval/Class Loss': class_loss.item(),
                'Eval/Seg Loss': seg_loss.item(),
                'Class Probs': class_probs,
                'Class Preds': class_preds,
                'Class Labels': class_labels,
                'Seg Probs': seg_probs,
                'Seg Preds': seg_preds,
                'Seg Labels': seg_labels,
            })
        else:
            metrics.update({
                'Train/Loss': loss.item(),
                'Train/Class Loss': class_loss.item(),
                'Train/Seg Loss': seg_loss.item()
            })
        return loss, metrics

    def process_eval_metrics(self, metrics):
        avg_loss = sum(metrics['Eval/Loss']) / len(metrics['Eval/Loss'])
        # class
        class_preds = np.array([pred for sublist in metrics['Class Preds']
                          for pred in sublist])
        class_labels = np.array([label for sublist in metrics['Class Labels']
                           for label in sublist])
        correct = np.equal(class_preds, class_labels)
        class_avg_acc = sum(correct) / len(correct)
        # seg            
        seg_preds = np.array([pred for sublist in metrics['Seg Preds']
                          for pred in sublist])
        seg_labels = np.array([label for sublist in metrics['Seg Labels']
                           for label in sublist])
        correct = np.equal(seg_preds, seg_labels)
        seg_avg_acc = sum(correct) / len(correct)            
        eval_metrics = {'Eval/Loss': avg_loss,
                        'Eval/Class Accuracy': class_avg_acc,
                        'Eval/Seg Accuracy': seg_avg_acc}
        seg_confusion_matrix = wandb.plot.confusion_matrix(
            y_true=seg_labels,
            preds=seg_preds,
            class_names=['Continue', 'Start'])
        class_confusion_matrix = wandb.plot.confusion_matrix(
            y_true=class_labels,
            preds=class_preds,
            class_names=de_num_to_type)
        eval_metrics.update({'Seg Confusion Matrix': seg_confusion_matrix})
        eval_metrics.update({'Class Confusion Matrix': class_confusion_matrix})
        return eval_metrics


    def make_segment_transformer_dataset(self, essay_dataset):
        print('Making Segment Transformer Dataset...')
        encoded_segments = []
        attention_masks = []
        ner_features = []
        labels = []
        for essay in tqdm.tqdm(essay_dataset):
            essay_ner_features, seg_lens, essay_labels = essay.segments
            input_ids, attention_mask = essay.segment_tokens
            encoded_segments.append(input_ids)
            attention_masks.append(attention_mask)
            ner_features.append(essay_ner_features)
            labels.append(essay_labels)
        ner_features = torch.cat(ner_features, dim=0)
        encoded_segments = torch.stack(encoded_segments, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0).unsqueeze(-1)
        dataset = TensorDataset(encoded_segments, ner_features, attention_masks, labels)
        print(f'Dataset created with {len(dataset)} samples')
        return dataset