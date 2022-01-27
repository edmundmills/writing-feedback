import torch

from core.predicter import *


class TestSegmentTokenizer:
    def test_valid(self, essay, pred_args):
        tokenizer = SegmentTokenizer(pred_args)
        segments = essay.d_elems_text
        encoded, attention_mask = tokenizer.encode(segments)
        assert(encoded.shape == (pred_args.num_ner_segments, tokenizer.max_seq_len + 1))
        assert(attention_mask.shape == (pred_args.num_ner_segments,))
        assert(torch.sum(attention_mask).item() == len(segments))


class TestSegmentNERProbs:
    def test_valid(self, ner_probs, pred_args):
        segments, seg_lengths = Predicter(pred_args).segment_ner_probs(ner_probs)
        start_probs = segments[0,:,0]
        class_probs = segments[0,1:-1,0]
        assert(segments.size() == (1, pred_args.num_ner_segments, 16))
        assert(abs(min(seg_lengths)) >= 1)
        assert(torch.max(start_probs).item() <= 1)


class TestMakeNERFeatureDataset:
    def test_valid(self, dataset_with_ner_probs, pred_args):
        predicter = Predicter(pred_args)
        feature_dataset = predicter.segment_essay_dataset(dataset_with_ner_probs, print_avg_grade=True)
        feature_dataset = predicter.make_ner_feature_dataset(feature_dataset)
        # ner_features
        assert(feature_dataset[0:1][0].size() == (1,
                                                  pred_args.num_ner_segments,
                                                  predicter.num_features))
        # attention_masks
        assert(feature_dataset[0:1][1].size() == (1,
                                                  pred_args.num_ner_segments))
        # labels
        assert(feature_dataset[0:1][2].size() == (1,
                                                  pred_args.num_ner_segments,
                                                  1))


class TestMakeSegmentTransformerDataset:
    def test_valid(self, dataset_with_ner_probs, pred_args):
        predicter = Predicter(pred_args)
        tokenizer = SegmentTokenizer(pred_args)
        feature_dataset = predicter.segment_essay_dataset(dataset_with_ner_probs)
        feature_dataset = tokenizer.make_segment_transformer_dataset(feature_dataset)
        # encoded_segments
        assert(feature_dataset[0:1][0].size() == (1,
                                                  pred_args.num_ner_segments,
                                                  tokenizer.max_seq_len + 1))
        # attention_masks
        assert(feature_dataset[0:1][1].size() == (1,
                                                  pred_args.num_ner_segments))
        # labels
        assert(feature_dataset[0:1][2].size() == (1,
                                                  pred_args.num_ner_segments,
                                                  1))


class TestClassifierForward:
    def test_valid(self, dataset_with_ner_probs, pred_args):
        predicter = Predicter(pred_args)
        feature_dataset = predicter.segment_essay_dataset(dataset_with_ner_probs, print_avg_grade=True)
        feature_dataset = predicter.make_ner_feature_dataset(feature_dataset)
        sample = feature_dataset[0:2][0].to(predicter.classifier.device)
        output = predicter.classifier(sample)
        assert(output.size() == (2, pred_args.num_ner_segments, predicter.classifier.num_outputs))