import torch

from core.essay import Prediction
from core.models.segmentation_agent import *


class TestSegmentationModel:
    def test_call(self, encoded_essay, encoded_preds, seg_agent):
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        output = seg_agent.model(encoded_text.to(seg_agent.device),
                                 encoded_preds.to(seg_agent.device),
                                 attention_mask.to(seg_agent.device))
        assert(output.cpu().size() == (1, seg_agent.action_space_dim))

class TestEncode:
    def test_valid(self, seg_agent, essay):
        tokenized = seg_agent.encode(essay.text)
        assert(tokenized['input_ids'].size() == (1, 1024))
        assert(tokenized['attention_mask'].size() == (1, 1024))

class TestAct:
    def test_valid(self, seg_agent, seg_env):
        state = seg_env.reset()
        action = seg_agent.act(state)
        assert(isinstance(action, Prediction))
