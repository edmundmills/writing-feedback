import numpy as np
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from core.dataset import EssayDataset
from core.predicter import Predicter
from utils.config import parse_args, get_config
from utils.render import plot_ner_output, EssayRenderer


if __name__ == '__main__':

    render = False
    score = True
    plot_segments = False
    num_essays = 100
    essay_id = None
    predictions = 'by_seg_label'

    args = parse_args()
    args = get_config('base', args)

    
    dataset = EssayDataset.load(args.baseline_ner_dataset_path)
    renderer = EssayRenderer()
    predicter = Predicter(args.predict)


    f_scores = []
    num_segments = []
    running_seg_lens = []
    num_labels = []
    clipped_segments = []
    for essay in (dataset[i] for i in range(num_essays)):
        if essay_id is not None:
            essay = dataset.get_by_id(essay_id)
        ner_probs = essay.ner_probs
        segments, segment_lens = predicter.segment_ner_probs(ner_probs)
        num_seg = sum([1 for seg_len in segment_lens if seg_len > 0])
        num_segments.append(num_seg)
        running_seg_lens.extend([seg_len for seg_len in segment_lens if seg_len > 0])
        clipped_segments.append(int(num_seg == args.predict.num_ner_segments))
        num_labels.append(len(essay.correct_predictions))
        if predictions == 'by_seg_label':
            seg_labels = essay.get_labels_for_segments(segment_lens)
            preds = essay.segment_labels_to_preds(seg_labels)
        elif predictions == 'by_heuristics':
            preds, _grade = predicter.by_heuristics(essay, thresholds=False)
        else:
            preds = None
        if score:
            grade = essay.grade(preds)
            f_scores.append(grade['f_score'])
        if render:
            renderer.render(essay, segment_lens=segment_lens, predictions=preds)
        if plot_segments:
            plot_ner_output(ner_probs, segment_lens=segment_lens)
            plot_ner_output(segments)
        if essay_id is not None:
            break
    if score:
        print(f'Avg F-score:\t\t{sum(f_scores) / len(f_scores):.2f}')
        print(f'Avg Num Segments:\t{sum(num_segments) / len(num_segments)}')
        print(f'Avg Num Labels:\t\t{sum(num_labels) / len(num_labels)}')
        print(f'Avg Segment Length:\t{sum(running_seg_lens) / len(running_seg_lens):.2f}')
        print(f'Frac. w/ Max Segments:\t{sum(clipped_segments) / len(clipped_segments):.2f}')