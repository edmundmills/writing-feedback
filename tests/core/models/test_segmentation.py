import torch

from core.models.segmentation import *


class TestSegmentationModel:
    def test_forward(self, encoded_essay, seg_args):
        model = SegmentationModel(seg_args)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        x = torch.stack((encoded_text, attention_mask), dim=1)
        output = model(x)
        assert(output.size() == (1, seg_args.essay_max_tokens))

class TestEncode:
    def test_valid(self, seg_args, essay):
        tokenizer = SegmentationTokenizer(seg_args)
        tokenized = tokenizer.encode(essay.text)
        assert(tokenized['input_ids'].size() == (1, 1024))
        assert(tokenized['attention_mask'].size() == (1, 1024))

class TestAgent:
    def test_make(self, seg_env, seg_args):
        agent = make_agent(seg_args, seg_env)
