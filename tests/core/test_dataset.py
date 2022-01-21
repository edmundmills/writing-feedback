from pathlib import Path

import numpy as np
import pandas as pd
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





