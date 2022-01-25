import numpy as np
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.essay import Prediction
from core.dataset import EssayDataset
from core.predicter import Predicter
from utils.config import parse_args, get_config
from utils.render import plot_ner_output, EssayRenderer


if __name__ == '__main__':
    args = parse_args()
    args = get_config('base', args)

    
    dataset = EssayDataset.load(args.ner_dataset_path)
    renderer = EssayRenderer()
    predicter = Predicter()

    num_essays=3

    for essay in (dataset[i] for i in range(num_essays)):
        # print(essay.essay_id, len(essay.words))
        # for pred in essay.correct_predictions:
        #     print(pred)
        ner_probs = essay.ner_probs
        segments = predicter.segment_ner_probs(ner_probs)
        segment_lens = segments[:,:,-1].squeeze().tolist()
        # preds = []
        # word_idx = 0
        # for length in segment_lens:
        #     if length < 1: continue
        #     pred = Prediction(word_idx, word_idx + length - 1, 0, essay.essay_id)
        #     word_idx += length
        #     preds.append(pred)
        # seg_labels = essay.get_labels_for_segments(preds)
        # seg_preds = essay.segment_labels_to_preds(seg_labels)
        preds, _grade = predicter.by_heuristics(essay, thresholds=False)
        renderer.render(essay, segment_lens=segment_lens, predictions=preds)
        # plot_ner_output(ner_probs, segment_lens=segment_lens)
        # plot_ner_output(segments)