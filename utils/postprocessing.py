import numpy as np
import pandas as pd

from core.essay import Prediction

def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

def link_evidence(oof):
    """
    https://www.kaggle.com/abhishek/two-longformers-are-better-than-1
    """
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(26,27, 1):
        retval = []
        for idv in idu:
            for c in  ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
                   'Counterclaim', 'Rebuttal']:
                q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i,r in q.iterrows():
                    pst = pst +[-1] + [int(x) for x in r['predictionstring'].split()]
                start = 1
                end = 1
                for i in range(2,len(pst)):
                    cur = pst[i]
                    end = i
                    #if pst[start] == 205:
                    #   print(cur, pst[start], cur - pst[start])
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and ((pst[i+1] > pst[end-1] + thresh) or (pst[i+1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end+1))
                #print(v)
                retval.append(v)
        roof = pd.DataFrame(retval, columns = ['id', 'class', 'predictionstring']) 
        roof = roof.merge(neoof, how='outer')
        return roof

proba_thresh = {
    "Lead": 0.7,
    "Position": 0.55,
    "Evidence": 0.65,
    "Claim": 0.55,
    "Concluding Statement": 0.7,
    "Counterclaim": 0.5,
    "Rebuttal": 0.55,
    'None': 1,
}
min_thresh = {
    "Lead": 9,
    "Position": 5,
    "Evidence": 14,
    "Claim": 3,
    "Concluding Statement": 11,
    "Counterclaim": 6,
    "Rebuttal": 4,
    'None': -1
}

def assign_by_heuristics(self, essay, thresholds=True): 
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
        if (class_confidence > proba_thresh[pred.argument_name] \
                and len(pred) > min_thresh[pred.argument_name]) \
                    or not thresholds:
            predictions.append(pred)
        cur_pred_class = pred_class
        cur_pred_start = idx
    pred = Prediction(cur_pred_start, idx, cur_pred_class, essay.essay_id)
    pred_weights = pred_probs[pred.start:(pred.stop + 1)]
    class_confidence = sum(pred_weights) / len(pred_weights)
    if (class_confidence > proba_thresh[pred.argument_name] \
            and len(pred) > min_thresh[pred.argument_name]) \
                or not thresholds:
        predictions.append(pred)
    metrics = essay.grade(predictions)
    return predictions, metrics
