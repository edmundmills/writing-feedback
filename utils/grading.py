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
