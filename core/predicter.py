from typing import Union

import torch
import torch.nn.functional as F
import numpy as np

from core.essay import Prediction


class Predicter:
    def __init__(self) -> None:
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


    def segment_ner_probs(self, ner_probs:Union[torch.Tensor, np.ndarray], max_segments=32):
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
        target_segments = max_segments
        result_segments = 0
        while result_segments < max_segments:
            threshold, _ = torch.kthvalue(start_probs, num_words - target_segments + 1, dim=1)
            threshold = threshold.item()
            segments = start_probs > threshold
            segments[:,0,:] = True
            for i in range(1, num_words):
                if (segments[:,i-1,:] or segments[:,i-2,:]) and segments[:,i,:]:
                    segments[:,i,:] = False
            result_segments = torch.sum(segments).item()
            target_segments += max_segments - result_segments
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
        padding = max(0, max_segments - n_segments)
        segmented = torch.cat((segmented[:max_segments], -torch.ones((padding, 16))), dim=0)
        segmented = segmented.unsqueeze(0)
        segment_lens = segmented[:,:,-1].squeeze().tolist()
        return segmented, segment_lens
