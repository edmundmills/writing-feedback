import torch

from core.segmentation import *
from core.env import SegmentationEnv

class TestSegmentNERProbs:
    def test_valid(self, ner_probs):
        segments = segment_ner_probs(ner_probs)
        seg_lengths = segments[0,:,-1]
        start_probs = segments[0,:,0]
        class_probs = segments[0,1:-1,0]
        assert(segments.size() == (1, 32, 10))
        assert(abs(torch.min(seg_lengths).item()) >= 1)
        assert(torch.max(start_probs).item() <= 1)

class TestAgent:
    def test_make(self, dataset_with_ner_probs, base_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, base_args.env)
        make_agent(base_args, env)
