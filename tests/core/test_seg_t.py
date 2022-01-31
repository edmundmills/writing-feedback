from core.segment_transformer import *

class TestModel:
    def test_make_join_dataset(self, seg_t_args, dataset_with_segments):
        model = SegmentTransformer(seg_t_args)
        dataset = model.make_join_dataset(dataset_with_segments)

    def test_make_polarity_dataset(self, seg_t_args, dataset_with_segments):
        model = SegmentTransformer(seg_t_args)
        dataset = model.make_polarity_dataset(dataset_with_segments)

    def test_make_classification_dataset(self, seg_t_args, dataset_with_segments):
        model = SegmentTransformer(seg_t_args)
        dataset = model.make_classification_dataset(dataset_with_segments)
