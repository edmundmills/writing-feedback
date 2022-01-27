import itertools
import math
from typing import Dict, List, Tuple

import numpy as np

from utils.constants import essay_dir, de_num_to_type, de_type_to_num
from utils.pstrings import ismatch, prediction_string, start_num, end_num


class Prediction:
    def __init__(self, start, stop, label, essay_id) -> None:
        self.start = int(start)
        self.stop = int(stop)
        if stop < start:
            raise ValueError('Prediction start cant be before prediction stop')
        self.label = label
        self.essay_id = essay_id

    def __len__(self):
        return len(self.word_idxs)

    @property
    def word_idxs(self):
        return list(range(self.start, self.stop + 1))

    @property
    def pstring(self):
        return ' '.join(str(num) for num in self.word_idxs)

    @property
    def argument_name(self):
        return de_num_to_type[self.label]

    def formatted(self):
        return {'id': self.essay_id,
                'class': de_num_to_type[self.label],
                'predictionstring': self.pstring}

    def __repr__(self):
        return str(self.formatted())
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        return bool(self.start == __o.start and self.stop == __o.stop
                    and self.label == __o.label and self.essay_id == __o.essay_id)


class Essay:
    def __init__(self, essay_id, text, labels, ner_probs=None, fold=None) -> None:
        self.essay_id = essay_id
        self.text = text
        self.labels = labels
        self.ner_probs = ner_probs
        self.fold = fold

    def __len__(self):
        return len(self.words)

    @property
    def path(self):
        return essay_dir / f'{self.essay_id}.txt'

    def _get_label(self, pstring):
        for label, labelname in enumerate(de_num_to_type):
            prediction = {'id': self.essay_id,
                        'class': labelname,
                        'predictionstring': pstring}
            grading_data = self.grade([prediction])
            if grading_data['true_positives'] == 1:
                return label
        return 0

    def ner_labels(self, num_words, predictions=None):
        predictions = predictions or self.correct_predictions
        tokens = [-1] * num_words
        for pred in predictions:
            start_token = pred.label
            cont_token = start_token + 7 if start_token != 0 else 0
            if pred.start < num_words:
                tokens[pred.start] = start_token
                stop_idx = min(num_words, pred.stop + 1)
                tokens[(pred.start+1):(stop_idx)] = [cont_token] * (len(pred) - 1)
        tokens = np.array(tokens, dtype=np.int8)
        return tokens

    def get_labels(self, predictions, num_d_elems=None):
        if not isinstance(predictions, list):
            predictions = [predictions]
        if len(predictions) > 0 and isinstance(predictions[0], Prediction):
            pstrings = (pred.pstring for pred in predictions)
        else:
            pstrings = predictions
        labels = [self._get_label(pstring) for pstring in pstrings]
        if num_d_elems:
            labels = labels[:num_d_elems] + [-1] * max(0, num_d_elems - len(labels))
        return labels

    def get_labels_for_segments(self, segment_lens:List[int]) -> List[Tuple[int, int]]:
        num_labels = len(segment_lens)
        seg_labels = [0] * num_labels
        matched_label_idxs = [None] * num_labels

        word_indices = np.arange(len(self))
        label_arrays = []
        for _text, d_type, pstring in self._all_d_elems():
            if d_type == 0: next
            label_array = (word_indices >= start_num(pstring)) * (word_indices <= end_num(pstring))
            label_arrays.append((label_array, de_type_to_num[d_type]))

        right_borders = list(itertools.accumulate(seg_len for seg_len in segment_lens if seg_len > 0))
        left_border = 0

        for idx, right_border in enumerate(right_borders):
            seg_array = (word_indices >= left_border) * (word_indices < right_border)
            for label_idx, (label_array, d_type) in enumerate(label_arrays):
                overlap = sum(seg_array * label_array)
                if overlap >= math.ceil((right_border - left_border) / 2):
                    seg_labels[idx] = d_type
                    matched_label_idxs[idx] = label_idx
                    break
            left_border = right_border
        for idx in range(len(seg_labels)):
            if idx == 0: continue
            if matched_label_idxs[idx - 1] == matched_label_idxs[idx]:
                if seg_labels[idx] != 0:
                    seg_labels[idx] += 7
        return list(zip(segment_lens, seg_labels))
    
    def segment_labels_to_preds(self, seg_labels) -> List[Prediction]:
        preds = []
        for idx, (seg_len, seg_label) in enumerate(seg_labels):
            if seg_len <= 0: continue
            if idx == 0:
                cur_start = 0
                cur_stop = seg_len
                cur_label = seg_label if seg_label <= 8 else seg_label - 7
            elif cur_label != 0 and seg_label - 7 == cur_label:
                cur_stop = cur_stop + seg_len
            else:
                preds.append(Prediction(cur_start, cur_stop - 1, cur_label, self.essay_id))
                cur_start = cur_stop
                cur_stop = cur_start + seg_len
                cur_label = seg_label if seg_label <= 8 else seg_label - 7
        preds.append(Prediction(cur_start, cur_stop - 1, cur_label, self.essay_id))
        return preds

    def grade(self, predictions:List[Dict]):
        if not isinstance(predictions, list):
            predictions = list(predictions)
        if predictions and isinstance(predictions[0], Prediction):
            predictions = [pred.formatted() for pred in predictions]
        predictions = [prediction for prediction in predictions
                       if prediction['class'] != 'None']
        matched_labels = [0] * len(self.labels)
        for prediction in predictions:
            for idx, (_, label) in enumerate(self.labels.iterrows()):
                if ismatch(prediction, label):
                    matched_labels[idx] = 1
                    break
        true_positives = sum(matched_labels)
        false_positives = len(predictions) - true_positives
        false_negatives = len(self.labels) - true_positives
        f_score = true_positives / (true_positives + false_positives + false_negatives)
        return {
            'f_score': f_score,
            'true_positives': true_positives,
            'false_negatives': false_negatives,
            'false_positives': false_positives,
        }

    @property
    def words(self) -> List[str]:
        return self.text.split()

    @property
    def d_elems_text(self) -> List[str]:
        return self.labels.loc[:,'discourse_text'].tolist()

    @property
    def pstrings(self) -> List[str]:
        return self.labels.loc[:,'predictionstring'].tolist()

    @property
    def correct_predictions(self) -> List[Prediction]:
        preds = [Prediction(start_num(pstring), end_num(pstring),
                            de_type_to_num[d_type], self.essay_id)
                 for _text, d_type, pstring in self._all_d_elems()]
        return preds            

    def _all_d_elems(self) -> List[Tuple]:
        arguments = []
        word_idx = 0
        for _, row in self.labels.iterrows():
            word_idxs = [int(num) for num in row.loc['predictionstring'].split()]
            if word_idxs[0] > word_idx:
                sentence = ' '.join(self.words[word_idx:word_idxs[0]])
                arguments.append((sentence, 'None', prediction_string(word_idx, word_idxs[0] - 1)))
            arguments.append((row.loc['discourse_text'], row.loc['discourse_type'], row.loc['predictionstring']))
            word_idx = word_idxs[-1] + 1
        if len(self.words) < word_idx:
            arguments.append((self.words[word_idx:], 'None', prediction_string(word_idx, len(self.words) + 1)))
        return arguments

    def random_pstrings(self, max_d_elems=None) -> List[str]:
        pstrings = []
        i = 0
        num_words = len(self.words)
        while i < num_words and (max_d_elems is None or len(pstrings) < max_d_elems):
            n = int(np.random.poisson(lam=3) * 15/3)
            n = max(n, 5)
            pstring = prediction_string(i, min(i+n, num_words - 1))
            pstrings.append(pstring)
            i += n + 1
        return pstrings