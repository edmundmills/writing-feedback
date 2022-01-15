import torch

from core.segmentation import *
from core.ner import NERModel

class TestNERModel:
    def test_forward_as_feature_extractor(self, encoded_essay, base_args):
        model = NERModel(base_args.ner, feature_extractor=True)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        x = torch.stack((encoded_text, attention_mask), dim=1).to(model.device)
        output = model(x)
        assert(output.size() == (1, base_args.ner.essay_max_tokens))

class TestAgent:
    def test_make(self, seq_env, base_args):
        make_agent(base_args, seq_env)
