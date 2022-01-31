from core.prediction import *
from core.segmenter import Segmenter


class TestMakeNERFeatureDataset:
    def test_valid(self, dataset_with_ner_probs, seg_args, pred_args):
        segmenter = Segmenter(seg_args)
        feature_dataset = segmenter.segment_essay_dataset(dataset_with_ner_probs, print_avg_grade=True)
        feature_dataset = Predicter(pred_args).make_ner_feature_dataset(feature_dataset)
        # ner_features
        assert(feature_dataset[0:1][0].size() == (1,
                                                  seg_args.num_ner_segments,
                                                  segmenter.num_features))
        # attention_masks
        assert(feature_dataset[0:1][1].size() == (1,
                                                  seg_args.num_ner_segments))
        # labels
        assert(feature_dataset[0:1][2].size() == (1,
                                                  seg_args.num_ner_segments,
                                                  1))


class TestClassifierForward:
    def test_valid(self, dataset_with_ner_probs, pred_args, seg_args):
        segmenter = Segmenter(seg_args)
        feature_dataset = segmenter.segment_essay_dataset(dataset_with_ner_probs, print_avg_grade=True)
        predicter = Predicter(pred_args)
        feature_dataset = predicter.make_ner_feature_dataset(feature_dataset)
        sample = feature_dataset[0:2][0].to(predicter.device)
        output = predicter(sample)
        assert(output.size() == (2, pred_args.num_ner_segments, predicter.num_outputs))