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

def get_labels(pstrings, essay, num_d_elems=None):
    labels = [get_label(pstring, essay) for pstring in pstrings]
    if num_d_elems:
        labels = labels[:num_d_elems] + [-1] * max(0, num_d_elems - len(labels))
    return labels

def to_tokens(predictions, num_words):
    tokens = []
    for pred in predictions:
        tokens.append(1)
        tokens.extend([0] * (len(pred.word_idxs) - 1))
    while len(tokens) < num_words:
        tokens.append(-1)
    return np.array(tokens[:num_words], dtype=np.int8)

def start_num(pstring):
    return int(pstring.split()[0])

def end_num(pstring):
    return int(pstring.split()[-1])