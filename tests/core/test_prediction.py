from core.prediction import *
from core.segmenter import Segmenter


class TestMakeNERFeatureDataset:
    def test_valid(self, tokenized_dataset, base_args):
        feature_dataset = Predicter(base_args.predict).make_dataset(tokenized_dataset, base_args)
        # ner_features
        assert(feature_dataset[0:1][0].size() == (1,
                                                  base_args.seg.num_ner_segments,
                                                  base_args.predict.feature_dim))
        # attention_masks
        assert(feature_dataset[0:1][1].size() == (1,
                                                  base_args.seg.num_ner_segments))
        # labels
        assert(feature_dataset[0:1][2].size() == (1,
                                                  base_args.seg.num_ner_segments,
                                                  1))


class TestClassifierForward:
    def test_valid(self, tokenized_dataset, base_args):
        predicter = Predicter(base_args.predict)
        feature_dataset = predicter.make_dataset(tokenized_dataset, base_args)
        sample = feature_dataset[0:2][0].to(predicter.device)
        output = predicter(sample)
        assert(output.size() == (2, base_args.seg.num_ner_segments, predicter.num_outputs))