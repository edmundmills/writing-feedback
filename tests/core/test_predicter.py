import torch

from core.predicter import *

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
        assert(feature_dataset[0:1][0].size() == (1,
                                                  pred_args.num_ner_segments,
                                                  predicter.num_features))
        assert(feature_dataset[0:1][1].size() == (1,
                                                  pred_args.num_ner_segments,
                                                  1))

class TestClassifierForward:
    def test_valid(self, dataset_with_ner_probs, pred_args):
        predicter = Predicter(pred_args)
        feature_dataset = predicter.segment_essay_dataset(dataset_with_ner_probs, print_avg_grade=True)
        feature_dataset = predicter.make_ner_feature_dataset(feature_dataset)
        sample = feature_dataset[0:1][0].to(predicter.classifier.device)
        output = predicter.classifier(sample)
        assert(output.size() == (1, pred_args.num_ner_segments, predicter.classifier.num_outputs))