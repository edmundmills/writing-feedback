import numpy as np
import pickle
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from utils.config import parse_args, get_config
from core.predicter import Predicter
from utils.render import EssayRenderer

filtered_tokens = set(['þ', '\x94', '¨', '\x91', '\x97'])

def clean_word(word):
    if word[0] in filtered_tokens:
        return word[1:]
    else:
        return word

def collate_offset(seq, text, offset_mapping):
    token_idx = 0
    ner_indices = []
    for idx, word in enumerate(text.split()):
        c_word = clean_word(word)          
        if c_word == '':
            ner_indices.append(token_idx)
            token_idx += 1
            continue
        word_start, stop = offset_mapping[token_idx]
        partial_str = text[word_start:stop]
        while len(partial_str.split()) == 0 or partial_str in filtered_tokens:
            token_idx += 1
            word_start, stop = offset_mapping[token_idx]
            partial_str = text[word_start:stop]
        ner_indices.append(token_idx)
        while not (partial_str.split()[0] == c_word):
            if len(partial_str.split()) > 1:
                print(repr(partial_str), repr(c_word), repr(word))
                raise RuntimeError
            token_idx += 1
            _, stop = offset_mapping[token_idx]
            partial_str = text[word_start:stop]
        token_idx += 1
    ner_indices = np.array(ner_indices) + 1
    return seq[ner_indices]

def collate_columns(ner_probs):
    column_idxs = [14, 0, 2, 6, 10, 12, 4, 8, 1, 3, 7, 11, 13, 5, 9]
    return ner_probs[:,column_idxs]

if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)
    
    dataset = EssayDataset.load(args.dataset_path)
    
    with open('data/baseline_ner_probs.pkl', 'rb') as filename:
        samples = pickle.load(filename)

    renderer = EssayRenderer()
    predicter = Predicter()

    ner_probs = {}

    predictions = 'by_heuristics'

    grades = []
    for _k, v in samples.items():
        for sample in v:
            collated_ner = collate_offset(sample['raw_scores'], sample['text'], sample['offset_mapping'])
            collated_ner = collate_columns(collated_ner)
            collated_ner = torch.from_numpy(collated_ner).unsqueeze(0).float()
            dataset.ner_probs.update({sample['id']: collated_ner})
    #         essay = dataset.get_by_id(sample['id'])
    #         _, seg_lens = predicter.segment_ner_probs(essay.ner_probs)
    #         if predictions == 'by_seg_label':
    #             seg_labels = essay.get_labels_for_segments(seg_lens)
    #             preds = essay.segment_labels_to_preds(seg_labels)
    #         elif predictions == 'by_heuristics':
    #             preds, _grade = predicter.by_heuristics(essay, thresholds=True)
    #         else:
    #             preds = None
    #         renderer.render(essay, predictions=preds, segment_lens=seg_lens)
    #         grade = essay.grade(preds)
    #         grades.append(grade['f_score'])
    # avg_f_score = sum(grades) / len(grades)
    # print(avg_f_score)
    print(len(dataset.ner_probs))
    dataset.save('data/baseline_ner_dataset.pkl')

