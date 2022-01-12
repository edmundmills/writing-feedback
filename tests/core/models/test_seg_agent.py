import torch

from core.models.segmentation_agent import *

class TestPstringsToTokens:
    def test_valid(self, prediction):
        length = 50
        tokens = to_tokens([prediction], length)
        print(tokens)
        assert(len(tokens) == length)
        assert(set(tokens) == set(['MASK', 'CONT', 'START']))

class TestEncode:
    def test_valid(self, seg_agent, essay):
        encoded = seg_agent.encode(essay.text)
        assert(isinstance(encoded, torch.Tensor))

class TestAct:
    def test_valid(self, seg_agent, seg_env):
        state = seg_env.reset()
        action = seg_agent.act(state)
        assert(isinstance(action, torch.Tensor))
        assert(action.size() == (seg_agent.action_space_dim,))