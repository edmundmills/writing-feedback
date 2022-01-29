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


class NERClassifier(Model):
    def __init__(self, pred_args):
        super().__init__()
        d_model = len(ner_num_to_token) + 1
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