import torch

from core.predicter import *

class TestSegmentNERProbs:
    def test_valid(self, ner_probs):
        segments = Predicter().segment_ner_probs(ner_probs)
        seg_lengths = segments[0,:,-1]
        start_probs = segments[0,:,0]
        class_probs = segments[0,1:-1,0]
        assert(segments.size() == (1, 32, 16))
        assert(abs(torch.min(seg_lengths).item()) >= 1)
        assert(torch.max(start_probs).item() <= 1)

