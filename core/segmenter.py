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
from utils.render import plot_ner_output

class Segmenter:
    def __init__(self, seg_args) -> None:
        self.num_ner_segments = seg_args.num_ner_segments
        self.seg_thresh = seg_args.seg_confidence_thresh
        self.num_features = len(ner_num_to_token) + 1


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
            start_probs = seg_data[0][:3,1:7]
            start_probs = torch.sum(start_probs, dim=0, keepdim=True) / min(3, start_probs.size(0))
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
        # plot_ner_output(segmented)
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
            essay_dataset.essays[essay.essay_id]['segments'] = Segments(ner_features, segment_lens, essay_labels)
        print('Dataset Segmented')
        if print_avg_grade:
            grade = sum(scores) / len(scores)
            print(f'Average Maximum Possible Grade: {grade}')
        return essay_dataset