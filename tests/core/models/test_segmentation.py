import torch

from core.models.segmentation import *


class TestSegmentationModel:
    def test_forward_as_feature_extractor(self, encoded_essay, seg_args):
        model = SegmentationModel(seg_args.ner, feature_extractor=True)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        x = torch.stack((encoded_text, attention_mask), dim=1).to(model.device)
        output = model(x)
        assert(output.size() == (1, seg_args.ner.essay_max_tokens))

    def test_forward_ner(self, encoded_essay, seg_args):
        model = SegmentationModel(seg_args.ner)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        output = model(encoded_text.to(model.device),
                       attention_mask.to(model.device))
        assert(output.size() == (1, seg_args.ner.essay_max_tokens, 2))

class TestEncode:
    def test_valid(self, seg_args, essay):
        tokenizer = SegmentationTokenizer(seg_args.ner)
        tokenized = tokenizer.encode(essay.text)
        assert(tokenized['input_ids'].size() == (1, 1024))
        assert(tokenized['attention_mask'].size() == (1, 1024))

class TestAgent:
    def test_make(self, seg_env, seg_args):
        make_agent(seg_args, seg_env)
