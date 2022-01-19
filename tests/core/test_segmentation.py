import torch

from core.segmentation import *
from core.ner import NERModel


class TestAgent:
    def test_make(self, seq_env, base_args):
        make_agent(base_args, seq_env)
