from typing import Dict, List, Tuple

import numpy as np

from core.constants import essay_dir, argument_names
from utils.grading import get_labels, ismatch, prediction_string

class Prediction:
    def __init__(self, start, stop, label, essay_id) -> None:
        self.start = start
        self.stop = stop
        self.label = label
        self.essay_id = essay_id

    @property
    def word_idxs(self):
        return list(range(self.start, self.stop + 1))

    @property
    def pstring(self):
        return ' '.join(str(num) for num in self.word_idxs)

    @property
    def argument_name(self):
        return argument_names[self.label]

    def formatted(self):
        return {'id': self.essay_id,
                'class': argument_names[self.label],
                'predictionstring': self.pstring}

class Essay:
    def __init__(self, essay_id, text, labels) -> None:
        self.essay_id = essay_id
        self.text = text
        self.labels = labels

    @property
    def path(self):
        return essay_dir / f'{self.essay_id}.txt'

    def get_labels(self, predictions):
        return get_labels([pred.pstring for pred in predictions], self)

    def grade(self, predictions:List[Dict]):
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

    def all_arguments(self) -> List[Tuple]:
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

    def polarity_pairs(self):
        text_pairs = []
        labels = []
        essay_arguments = self.labels[['discourse_type', 'discourse_text']].values.tolist()
        lead = None
        position = None
        conclusion = None
        claims = []
        counterclaims = []
        evidences = []
        prev_arg = None
        for arg_type, arg_text in essay_arguments:
            if arg_type == 'Lead':
                lead = arg_text
            elif arg_type == 'Position':
                position = arg_text
            elif arg_type == 'Concluding Statement':
                conclusion = arg_text
            elif arg_type == 'Claim':
                claims.append(arg_text)
            elif arg_type == 'Counterclaim':
                counterclaims.append(arg_text)
            elif arg_type == 'Evidence':
                evidences.append(arg_text)
            if prev_arg is not None:
                if prev_arg[0] == 'Claim' and arg_type == 'Evidence':
                    text_pairs.append((prev_arg[1], arg_text))
                    labels.append(1)
                # elif prev_arg[0] == 'Claim' and arg_type == 'Counterclaim':
                #     text_pairs.append((prev_arg[1], arg_text))
                #     labels.append(-1)
                elif prev_arg[0] == 'Counterclaim' and arg_type == 'Rebuttal':
                    text_pairs.append((prev_arg[1], arg_text))
                    labels.append(-1)
            prev_arg = arg_type, arg_text
        if position:
            text_pairs.extend(((position, claim) for claim in claims))
            labels.extend(1 for _ in claims)
            # text_pairs.extend(((evidence, position) for evidence in evidences))
            # labels.extend(0 for _ in evidences)
            text_pairs.extend(((position, claim) for claim in counterclaims))
            labels.extend(-1 for _ in counterclaims)
        if conclusion:
            # text_pairs.extend(((conclusion, claim) for claim in claims))
            # labels.extend(1 for _ in claims)
            text_pairs.extend(((conclusion, claim) for claim in counterclaims))
            labels.extend(-1 for _ in counterclaims)
            text_pairs.extend(((evidence, conclusion) for evidence in evidences))
            labels.extend(0 for _ in evidences)
        # if evidences and len(evidences) >= 2:
        #     text_pairs.extend(permutations(evidences, 2))
        #     labels.extend(0 for _ in permutations(evidences, 2))
        for counterclaim in counterclaims:
            text_pairs.extend((claim, counterclaim) for claim in claims)
            labels.extend(-1 for _ in claims)
            text_pairs.extend((counterclaim, claim) for claim in claims)
            labels.extend(-1 for _ in claims)
        if lead:
            text_pairs.extend(((evidence, lead) for evidence in evidences))
            labels.extend(0 for _ in evidences)
            # text_pairs.extend(((lead, evidence) for evidence in evidences))
            # labels.extend(0 for _ in evidences)
        return text_pairs, labels

    def random_pstrings(self, max_d_elems=None) -> List[str]:
        pstrings = []
        i = 0
        n_words = len(self.words)
        while i < n_words and (max_d_elems is None or len(pstrings) < max_d_elems):
            n = int(np.random.poisson(lam=3) * 15/3)
            n = max(n, 5)
            pstring = prediction_string(i, min(i+n, n_words - 1))
            pstrings.append(pstring)
            i += n + 1
        return pstrings