import torch

from core.segmenter import *


class TestSegmentNERProbs:
    def test_valid(self, dataset_with_ner_probs, seg_args):
        essay = dataset_with_ner_probs[0]
        segments, seg_lengths = Segmenter(seg_args).segment_ner_probs(essay.ner_probs)
        start_probs = segments[0,:,0]
        class_probs = segments[0,1:-1,0]
        assert(segments.size() == (1, seg_args.num_ner_segments, 16))
        assert(abs(min(seg_lengths)) >= 1)
        assert(torch.max(start_probs).item() <= 1)



