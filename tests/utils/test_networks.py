import torch.nn as nn

from utils.networks import *


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