from functools import partial
from typing import Dict, List

import pandas as pd

def ismatch(prediction, label):
    if prediction['class'] != label.loc['discourse_type']:
        return False
    pred_word_indices = set(int(num) for num in prediction['predictionstring'].split())
    label_word_indices = set(int(num) for num in label.loc['predictionstring'].split())
    match_word_indices = pred_word_indices & label_word_indices
    return len(match_word_indices) / len(label_word_indices) > 0.5 \
                    and len(match_word_indices) / len(pred_word_indices) > 0.5

def grade(feedback:List[Dict], essay:pd.DataFrame):
    matched_labels = [0] * len(essay)
    for prediction in feedback:
        for idx, (_, label) in enumerate(essay.iterrows()):
            if ismatch(prediction, label):
                matched_labels[idx] = 1
                break
    true_positives = sum(matched_labels)
    false_positives = len(feedback) - true_positives
    false_negatives = len(essay) - true_positives
    f_score = true_positives / (true_positives + false_positives + false_negatives)
    return {
        'f_score': f_score,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
    }
