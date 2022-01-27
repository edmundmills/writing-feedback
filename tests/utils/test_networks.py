import torch.nn as nn

from utils.networks import *


def test_positional_encoding():
    pos_encoder = PositionalEncoder(seq_len=20, features=768)
    input_tokens = torch.rand(20, 768)
    pos_encoded = pos_encoder(input_tokens)
    assert(pos_encoded.size() == input_tokens.size())
    input_tokens = input_tokens.unsqueeze(0)
    pos_encoded = pos_encoder(input_tokens)
    assert(pos_encoded.size() == input_tokens.size())


class TestMode:
    def test_on(self):
        module = nn.Linear(4,5)
        module.eval()
        with Mode(module, mode='train'):
            assert(module.training)
        assert(not module.training)

    def test_off(self):
        module = nn.Linear(4,5)
        module.train()
        with Mode(module, mode='eval'):
            assert(not module.training)
        assert(module.training)

    def test_on_on(self):
        module = nn.Linear(4,5)
        module.train()
        with Mode(module, mode='train'):
            assert(module.training)
        assert(module.training)

    def test_off_off(self):
        module = nn.Linear(4,5)
        module.eval()
        with Mode(module, mode='eval'):
            assert(not module.training)
        assert(not module.training)