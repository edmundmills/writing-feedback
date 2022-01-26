from collections import deque
import time
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
import wandb

from core.essay import Prediction
from utils.constants import ner_num_to_token
from utils.networks import Model, MLP, PositionalEncoder, Mode


class NERClassifier(Model):
    def __init__(self, pred_args):
        super().__init__()
        d_model = len(ner_num_to_token) + 1
        self.num_outputs = len(ner_num_to_token)
        self.seq_len = pred_args.num_ner_segments
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=pred_args.n_head,
                                                   batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer,
                                               pred_args.num_attention_layers).to(self.device)
        self.positional_encoder = PositionalEncoder(features=d_model,
                                                    seq_len=self.seq_len,
                                                    device=self.device)
        self.linear = MLP(n_inputs=d_model,
                          n_outputs=self.num_outputs,
                          n_layers=pred_args.num_linear_layers,
                          dropout=pred_args.dropout,
                          layer_size=pred_args.linear_layer_size).to(self.device)
    
    def forward(self, features):
        y = self.positional_encoder(features)
        y = self.attention(y)
        y = self.linear(y)
        return y

    def learn(self, train_dataset, val_dataset, args):
        print(len(train_dataset))
        base_args = args
        args = args.predict
        dataloader = DataLoader(train_dataset,
                                num_workers=4,
                                batch_size=args.batch_size,
                                )
        step = 0
        optimizer = torch.optim.AdamW(self.parameters(), args.learning_rate)
        running_loss = deque(maxlen=args.print_interval)
        timestamps = deque(maxlen=args.print_interval)

        with Mode(self, 'train'):
            for epoch in range(1, args.epochs + 1):
                print(f'Starting epoch {epoch}')
                for ner_features, labels in dataloader:
                    step += 1
                    ner_features = ner_features.to(self.device)
                    labels = labels.to(self.device)
                    output = self(ner_features)
                    output = output.reshape(-1, 15)
                    labels = labels.reshape(-1)

                    loss = F.cross_entropy(output, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss = loss.item()
                    running_loss.append(loss)
                    timestamps.append(time.time())
                    metrics = {
                        'Train Loss': loss,
                    }

                    if step % args.print_interval == 0:
                        print(f'Step {step}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
                            f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')

                    if step % args.eval_interval == 0:
                        eval_metrics = self.evaluate(val_dataset, base_args, n_samples=args.eval_samples)
                        metrics.update(eval_metrics)
                        print(f'Step {step}:\t{eval_metrics}')

                    if base_args.wandb:
                        wandb.log(metrics, step=step)
        print('Training Complete')



    def evaluate(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        base_args = args
        args = args.predict
        metrics = {}
        dataloader = DataLoader(dataset,
                        num_workers=4,
                        batch_size=args.batch_size)
        losses = []
        running_preds = []
        running_labels = []
        with Mode(self, 'eval'):
            for idx, (ner_features, labels) in enumerate(dataloader, start=1):
                ner_features = ner_features.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    output = self(ner_features)
                    output = output.reshape(-1, 15)
                    labels = labels.reshape(-1)
                    loss = F.cross_entropy(output, labels)
                losses.append(loss.item())
                probs = F.softmax(output, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1).flatten()
                running_preds.extend(preds)
                running_labels.extend(labels.cpu().numpy().flatten())
                if idx * args.batch_size >= n_samples:
                    break
        avg_loss = sum(losses) / len(losses)
        correct = np.equal(running_preds, running_labels)
        avg_acc = sum(correct) / len(running_preds)            
        metrics = {'Eval Loss': avg_loss,
                   'Eval Accuracy': avg_acc}
                
        if base_args.wandb:
            seg_confusion_matrix = wandb.plot.confusion_matrix(
                y_true=running_labels,
                preds=running_preds,
                class_names=ner_num_to_token)
            metrics.update({'Confusion Matrix': seg_confusion_matrix})
        return metrics



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
        target_segments = self.num_ner_segments
        result_segments = 0
        while result_segments < self.num_ner_segments:
            threshold, _ = torch.kthvalue(start_probs, num_words - target_segments + 1, dim=1)
            threshold = threshold.item()
            segments = start_probs > threshold
            segments[:,0,:] = True
            for i in range(1, num_words):
                if (segments[:,i-1,:] or segments[:,i-2,:]) and segments[:,i,:]:
                    segments[:,i,:] = False
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

    def make_ner_feature_dataset(self, essay_dataset, print_avg_grade=False):
        print('Making NER Feature Dataset...')
        scores = []
        features = []
        labels = []
        for essay in tqdm.tqdm(essay_dataset):
            ner_features, segment_lens = self.segment_ner_probs(essay.ner_probs)
            essay_labels = essay.get_labels_for_segments(segment_lens)
            if print_avg_grade:
                preds = essay.segment_labels_to_preds(essay_labels)
                score = essay.grade(preds)['f_score']
                scores.append(score)
            essay_labels = torch.LongTensor([seg_label for _, seg_label in essay_labels])
            features.append(ner_features)
            labels.append(essay_labels)
        features = torch.cat(features, dim=0)
        labels = torch.stack(labels, dim=0).unsqueeze(-1)
        dataset = TensorDataset(features, labels)
        print(f'Dataset created with {len(dataset)} samples')
        if print_avg_grade:
            grade = sum(scores) / len(scores)
            print(f'Average Maximum Possible Grade: {grade}')
        return dataset
