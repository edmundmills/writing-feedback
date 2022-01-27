from collections import Counter

import torch

from core.dataset import *


class TestEssayDataset:
    def test_init(self, dataset):
        assert(len(dataset) == 10)

    def test_idx(self, dataset):
        essay = dataset[0]
        assert(isinstance(essay, Essay))

    def test_random_essay(self, dataset):
        random_essay = dataset.random_essay(num_essays=5)
        assert(len(random_essay) == 5)
        assert(isinstance(random_essay[0], Essay))

    def test_random_span(self, dataset):
        random_span = dataset.random_span(num_words=5, num_spans=10)
        assert(len(random_span) == 10)
        assert(isinstance(random_span[0], str))
        assert(len(random_span[0].split()) == 5)

    def test_split_default(self, dataset):
        train, val = dataset.split()
        assert(isinstance(train, EssayDataset))
        assert(isinstance(val, EssayDataset))
        assert(len(train) == 9)
        assert(len(val) == 1)
        assert(set(train.df.loc[:,'id'].unique()) == set(train.essay_ids))
        assert(set(val.df.loc[:,'id'].unique()) == set(val.essay_ids))

    def test_split(self, dataset):
        datasets = dataset.split([.25, .25, .25, .25])
        assert(len(datasets) == 4)

    def test_split_one(self, dataset):
        datasets = dataset.split([1])
        new_dataset = datasets[0]
        assert(len(datasets) == 1)
        assert(len(new_dataset) == len(dataset))

    def test_set_ner_probs(self, dataset):
        assert(dataset.ner_probs == {})
        ner_probs = {'dsf': 'sdfgsdfg'}
        dataset.ner_probs = ner_probs
        assert(dataset.ner_probs == {})
        ner_probs = {k: 'test' for k in dataset.essay_ids}
        dataset.ner_probs = ner_probs
        assert(dataset.ner_probs == ner_probs)
        assert(dataset.ner_probs is not ner_probs)

    def test_set_segments(self, dataset):
        assert(dataset.segments == {})
        segments = {'dsf': 'sdfgsdfg'}
        dataset.segments = segments
        assert(dataset.segments == {})
        segments = {k: 'test' for k in dataset.essay_ids}
        dataset.segments = segments
        assert(dataset.segments == segments)
        assert(dataset.segments is not segments)

    def test_add_dataset(self, dataset_with_ner_probs):
        dataset = dataset_with_ner_probs
        datasets = dataset.split([.5,.5])
        new_dataset = datasets[0] + datasets[1]
        assert(len(new_dataset) == len(dataset))
        assert(new_dataset.essay_ids == dataset.essay_ids)
        essay_id = dataset.essay_ids[0]
        essay_1 = dataset.get_by_id(essay_id)
        essay_2 = new_dataset.get_by_id(essay_id)
        assert(torch.equal(essay_1.ner_probs, essay_2.ner_probs))

    def test_make_folds(self, dataset):
        dataset.make_folds(num_folds=4)
        counts = Counter()
        for essay in dataset:
            counts[essay.fold] += 1
        min_count, max_count = min(counts.values()), max(counts.values())
        assert(max_count - min_count <= 1)

    def test_get_folds(self, dataset):
        dataset.make_folds()
        fold = 0
        train, val = dataset.get_fold(fold)
        assert(len(train) == 8)
        assert(len(val) == 2)
        assert(all([essay.fold != fold for essay in train]))
        assert(all([essay.fold == fold for essay in val]))
        fold = 1
        train, val = dataset.get_fold(fold)
        assert(len(train) == 8)
        assert(len(val) == 2)
        assert(all([essay.fold != fold for essay in train]))
        assert(all([essay.fold == fold for essay in val]))