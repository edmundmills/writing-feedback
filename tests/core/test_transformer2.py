import torch

from core.transformer2 import *
from core.segmenter import Segmenter

class TestSegmentTokenizer:
    def test_valid(self, essay, seg_args):
        tokenizer = SegmentTokenizer(seg_args)
        segments = essay.d_elems_text
        encoded, attention_mask = tokenizer.encode(segments)
        assert(encoded.shape == (seg_args.num_ner_segments, tokenizer.max_seq_len))
        assert(attention_mask.shape == (seg_args.num_ner_segments,))
        assert(torch.sum(attention_mask).item() == len(segments))


class TestMakeSegmentTransformerDataset:
    def test_valid(self, dataset_with_ner_probs, seg_args):
        segmenter = Segmenter(seg_args)
        tokenizer = SegmentTokenizer(seg_args)
        feature_dataset = segmenter.segment_essay_dataset(dataset_with_ner_probs)
        feature_dataset = tokenizer.tokenize_dataset(feature_dataset)
        # encoded_segments
        assert(feature_dataset[0].segment_tokens[0].size() == (
                                                  seg_args.num_ner_segments,
                                                  tokenizer.max_seq_len))
        # attention_masks
        assert(feature_dataset[0].segment_tokens[1].size() == (
                                                  seg_args.num_ner_segments,))

