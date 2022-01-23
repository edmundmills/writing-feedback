from typing import Union

import torch
import numpy as np

from core.essay import Prediction


class Predicter:
    def __init__(self) -> None:
        self.start_thresh = 0.6

        self.proba_thresh = {
            "Lead": 0.7,
            "Position": 0.38,
            "Evidence": 0.55,
            "Claim": 0.6,
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

    def by_heuristics(self, essay): 
        ner_probs = essay.ner_probs
        start_preds = ner_probs[:,:,0].squeeze() > self.start_thresh
        class_probs = ner_probs[:,:,1:]
        class_preds = np.argmax(class_probs, axis=-1).squeeze()
        predictions = []
        for idx, (start_pred, class_pred) in enumerate(zip(start_preds, class_preds)):
            if idx == 0:
                cur_pred_start = 0
                cur_pred_class = class_preds[0]
                continue
            if class_pred == cur_pred_class and not start_pred:
                continue
            pred = Prediction(cur_pred_start, idx, int(cur_pred_class), essay.essay_id)
            pred_weights = class_probs[0, pred.start:(pred.stop + 1), pred.label]
            pred_weights = pred_weights.squeeze().tolist()
            class_confidence = sum(pred_weights) / len(pred_weights)
            if class_confidence > self.proba_thresh[pred.argument_name] \
                    and len(pred) > self.min_thresh[pred.argument_name]:
                predictions.append(pred.formatted())
            cur_pred_class = class_pred
            cur_pred_start = idx
        metrics = essay.grade(predictions)
        return predictions, metrics


    def segment_ner_probs(self, ner_probs:Union[torch.Tensor, np.ndarray], max_segments=32):
        ner_probs = torch.tensor(ner_probs)
        if len(ner_probs.size()) == 2:
            ner_probs = ner_probs.unsqueeze(0)
        num_words = ner_probs.size(1)
        threshold, _ = torch.kthvalue(ner_probs, num_words - max_segments + 1, dim=1)
        threshold = threshold[0,0]
        segments = ner_probs[0,:,0] > threshold
        segments = segments.tolist()
        segment_data = []
        cur_seg_data = []

        def concat_seg_data(seg_data):
            seg_len = len(seg_data)
            start_prob = seg_data[0][0,0]
            seg_data = torch.cat(seg_data, dim=0)
            seg_data = torch.sum(seg_data, dim=0, keepdim=True) / seg_len
            seg_data[:,0] = start_prob
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
        segmented = torch.cat((segmented[:max_segments], -torch.ones((padding, 10))), dim=0)
        segmented = segmented.unsqueeze(0)
        return segmented