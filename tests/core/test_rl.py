from core.agent import *
from core.env import SegmentationEnv


class TestAgent:
    def test_make(self, dataset_with_ner_probs, base_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, base_args)
        make_agent(base_args, env)
