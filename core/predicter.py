from collections import deque, namedtuple
import time
from typing import Union, List

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
import wandb

from core.dataset import SegmentTokens, Segments
from core.essay import Prediction
from utils.constants import ner_num_to_token
from utils.networks import Model, MLP, PositionalEncoder, Mode


class SegmentTokenizer(SentenceTransformer):
    def __init__(self, pred_args):
        print('Loading Segment Tokenizer...')
        super().__init__('sentence-transformers/all-mpnet-base-v2')
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


class NERClassifier(Model):
    def __init__(self, pred_args):
        super().__init__()
        if pred_args.name == 'Attention':
            d_model = len(ner_num_to_token) + 1
        elif pred_args.name == 'SentenceTransformer':
            d_model = 769
        self.num_outputs = len(ner_num_to_token)
        self.seq_len = pred_args.num_ner_segments
        self.linear = MLP(n_inputs=d_model,
                            n_outputs=pred_args.intermediate_layer_size,
                            n_layers=pred_args.num_linear_layers,
                            layer_size=pred_args.linear_layer_size,
                            dropout=pred_args.dropout).to(self.device)
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
    
    def forward(self, features):
        y = self.linear(features)
        y = self.positional_encoder(y)
        y = self.attention(y)
        y = self.head(y)
        return y

    def make_ner_feature_dataset(self, essay_dataset):
        print('Making NER Feature Dataset...')
        features = []
        labels = []
        attention_masks = []
        for essay in tqdm.tqdm(essay_dataset):
            ner_features, _seg_lens, essay_labels = essay.segments
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


class Predicter:
    def __init__(self, pred_args) -> None:
        self.start_thresh = 0.6
        self.proba_thresh = {
            "Lead": 0.7,
            "Position": 0.55,
            "Evidence": 0.65,
            "Claim": 0.55,
            "Concluding Statement": 0.7,
            "Counterclaim": 0.5,
            "Rebuttal": 0.55,
            'None': 1,
        }
        self.min_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4,
            'None': -1
        }
        self.num_ner_segments = pred_args.num_ner_segments
        self.seg_thresh = pred_args.seg_confidence_thresh
        self.classifier = NERClassifier(pred_args)
        self.num_features = len(ner_num_to_token) + 1



    def by_heuristics(self, essay, thresholds=True): 
        probs = essay.ner_probs.numpy()
        preds = np.argmax(probs, axis=-1).squeeze()
        pred_probs = np.max(probs, axis=-1).squeeze()
        predictions = []
        for idx, pred in enumerate(preds):
            start_pred = pred > 0 and pred <= 7
            pred_class = pred - 7 if pred > 7 else pred
            if idx == 0:
                cur_pred_start = 0
                cur_pred_class = pred_class
                continue
            if pred_class == cur_pred_class and not start_pred:
                continue
            pred = Prediction(cur_pred_start, idx - 1, cur_pred_class, essay.essay_id)
            pred_weights = pred_probs[pred.start:(pred.stop + 1)]
            class_confidence = sum(pred_weights) / len(pred_weights)
            if (class_confidence > self.proba_thresh[pred.argument_name] \
                    and len(pred) > self.min_thresh[pred.argument_name]) \
                        or not thresholds:
                predictions.append(pred)
            cur_pred_class = pred_class
            cur_pred_start = idx
        pred = Prediction(cur_pred_start, idx, cur_pred_class, essay.essay_id)
        pred_weights = pred_probs[pred.start:(pred.stop + 1)]
        class_confidence = sum(pred_weights) / len(pred_weights)
        if (class_confidence > self.proba_thresh[pred.argument_name] \
                and len(pred) > self.min_thresh[pred.argument_name]) \
                    or not thresholds:
            predictions.append(pred)
        metrics = essay.grade(predictions)
        return predictions, metrics


    def segment_ner_probs(self, ner_probs:Union[torch.Tensor, np.ndarray]):
        # ner_probs = torch.tensor(ner_probs)
        if len(ner_probs.size()) == 2:
            ner_probs = ner_probs.unsqueeze(0)
        num_words = torch.sum(ner_probs[:,:,0] != -1, dim=1).item()
        ner_probs = ner_probs[:,:num_words,:]
        
        start_probs = torch.sum(ner_probs[:,:,1:8], dim=-1, keepdim=True)

        ner_probs_offset = torch.cat((ner_probs[:,:1,:], ner_probs[:,:-1,:]), dim=1)
        delta_probs = ner_probs - ner_probs_offset
        max_delta = torch.max(delta_probs, dim=-1, keepdim=True).values
        start_probs += max_delta

        def remove_adjacent(segments):
            for i in range(1, segments.size(1)):
                if (segments[:,i-1,:] or segments[:,i-2,:]) and segments[:,i,:]:
                    segments[:,i,:] = False
            return segments

        segments = start_probs > self.seg_thresh
        segments[:,0,:] = True
        segments = remove_adjacent(segments)
        result_segments = torch.sum(segments).item()

        if result_segments > self.num_ner_segments:
            target_segments = result_segments
            while result_segments != self.num_ner_segments:
                threshold, _ = torch.kthvalue(start_probs, num_words - target_segments + 1, dim=1)
                threshold = threshold.item()
                segments = start_probs > threshold
                segments[:,0,:] = True
                segments = remove_adjacent(segments)
                result_segments = torch.sum(segments).item()
                target_segments += self.num_ner_segments - result_segments
        segments = segments.squeeze().tolist()
        segment_data = []
        cur_seg_data = []

        def concat_seg_data(seg_data):
            seg_len = len(seg_data)
            start_probs = seg_data[0][0,1:7]
            seg_data = torch.cat(seg_data, dim=0)
            seg_data = torch.sum(seg_data, dim=0, keepdim=True) / seg_len
            seg_data[:,1:7] = start_probs
            seg_data = torch.cat((seg_data, torch.tensor(seg_len).reshape(1,1)), dim=-1)
            return seg_data

        for div_idx, divider in enumerate(segments):
            if ner_probs[0, div_idx, 0].item() == -1:
                break
            if divider and cur_seg_data:
                cur_seg_data = concat_seg_data(cur_seg_data)
                segment_data.append(cur_seg_data)
                cur_seg_data = []
            cur_slice = ner_probs[:,div_idx]
            cur_seg_data.append(cur_slice)
        if cur_seg_data:
            cur_seg_data = concat_seg_data(cur_seg_data)
            segment_data.append(cur_seg_data)
        n_segments = len(segment_data)
        segmented = torch.cat(segment_data, dim=0)
        padding = max(0, self.num_ner_segments - n_segments)
        segmented = torch.cat((segmented[:self.num_ner_segments], -torch.ones((padding, 16))), dim=0)
        segmented = segmented.unsqueeze(0)
        segment_lens = segmented[:,:,-1].squeeze().tolist()
        return segmented, segment_lens

    def segment_essay_dataset(self, essay_dataset, print_avg_grade=False):
        print('Segmenting Dataset...')
        scores = []
        for essay in tqdm.tqdm(essay_dataset):
            ner_features, segment_lens = self.segment_ner_probs(essay.ner_probs)
            essay_labels = essay.get_labels_for_segments(segment_lens)
            if print_avg_grade:
                preds = essay.segment_labels_to_preds(essay_labels)
                score = essay.grade(preds)['f_score']
                scores.append(score)
            essay_labels = torch.LongTensor([seg_label for _, seg_label in essay_labels])
            essay_dataset[essay.essay_id]['segments'] = Segments(ner_features, segment_lens, essay_labels)
        print('Dataset Segmented')
        if print_avg_grade:
            grade = sum(scores) / len(scores)
            print(f'Average Maximum Possible Grade: {grade}')
        return essay_dataset