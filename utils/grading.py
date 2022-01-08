import numpy as np

from core.constants import argument_names

def prediction_string(start:int, stop:int):
    return ' '.join(str(num) for num in range(start, stop + 1))

def ismatch(prediction, label):
    if prediction['class'] != label.loc['discourse_type']:
        return False
    pred_word_indices = set(int(num) for num in prediction['predictionstring'].split())
    label_word_indices = set(int(num) for num in label.loc['predictionstring'].split())
    match_word_indices = pred_word_indices & label_word_indices
    return len(match_word_indices) / len(label_word_indices) > 0.5 \
                    and len(match_word_indices) / len(pred_word_indices) > 0.5

def to_predictions(predictionstrings, logits, essay_id):
    predictions = []
    for idx, predictionstring in enumerate(predictionstrings):
        label = np.argmax(logits[idx, ...].squeeze().numpy(), axis=0)
        if label == 0:
            continue
        prediction = {
            'id': essay_id,
            'class': argument_names[label],
            'predictionstring': predictionstring,
        }
        predictions.append(prediction)
    return predictions

def pstrings_to_tokens(predictionstrings, length):
    tokens = []
    for pstring in predictionstrings:
        word_idxs = [int(num) for num in pstring.split()]
        tokens.append('START')
        tokens.extend(['CONT'] * (len(word_idxs) - 1))
    while len(tokens) < length:
        tokens.append('MASK')
    return tokens

def get_discourse_elements(text, predictionstrings):
    txt_words = text.split()
    des = []
    for pstring in predictionstrings:
        de_words = [txt_words[int(num)] for num in pstring.split()]
        de = ' '.join(de_words)
        des.append(de)
    return des

def get_label(pstring, essay):
    for label, labelname in enumerate(argument_names):
        prediction = {'id': essay.essay_id,
                      'class': labelname,
                      'predictionstring': pstring}
        grading_data = essay.grade([prediction])
        if grading_data['true_positives'] == 1:
            return label
    return 0

def get_labels(pstrings, essay):
    return [get_label(pstring, essay) for pstring in pstrings]
        