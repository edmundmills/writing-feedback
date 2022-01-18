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

    def test_make_arg_class_dataset(self, fix_seed, dataset):
        class_dataset = dataset.make_arg_classification_dataset()
        assert(isinstance(class_dataset, ClassificationDataset))
        assert(isinstance(class_dataset[0][0], str))
        assert(isinstance(class_dataset[0][1], torch.Tensor))
        assert(len(class_dataset) == len(class_dataset.text))
        assert(len(class_dataset) == class_dataset.labels.size()[0])

    def test_make_balanced_arg_class_dataset(self, fix_seed, dataset):
        class_dataset = dataset.make_arg_classification_dataset(balanced=True)
        assert(isinstance(class_dataset, ClassificationDataset))
        assert(isinstance(class_dataset[0][0], str))
        assert(isinstance(class_dataset[0][1], torch.Tensor))
        assert(len(class_dataset) == len(class_dataset.text))
        assert(len(class_dataset) == class_dataset.labels.size()[0])
        assert(torch.sum(torch.eq(class_dataset.labels, 1)) == torch.sum(torch.eq(class_dataset.labels, 2)))

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

    def test_make_polarity_dataset(self, dataset):
        polarity_dataset = dataset.make_polarity_dataset()
        assert(isinstance(polarity_dataset, ComparisonDataset))
        assert(len(polarity_dataset) == len(polarity_dataset.text_pairs))
        assert(len(polarity_dataset) == polarity_dataset.labels.size()[0])
        assert(isinstance(polarity_dataset[0], tuple))
        assert(all(len(item) == 2 for item in polarity_dataset[:][0]))
        assert(all(type(item[0]) == str for item in polarity_dataset[:][0]))
        assert(isinstance(polarity_dataset[0][0], tuple))
        assert(isinstance(polarity_dataset[0][1], torch.Tensor))
        assert(isinstance(polarity_dataset[0][0][0], str))
        assert(isinstance(polarity_dataset[0][0][1], str))

    def test_make_ner_dataset(self, fix_seed, dataset, ner_tokenizer):
        essay_feedback_dataset = dataset.make_ner_dataset(tokenizer=ner_tokenizer)
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (1024,))
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (1024,))
        assert(isinstance(essay_feedback_dataset[0][2], torch.Tensor))
        assert(essay_feedback_dataset[0][2].size() == (1024,))
        assert(essay_feedback_dataset[0][2].max().item() == 1)
        assert(essay_feedback_dataset[0][2].min().item() == -1)

    def test_make_ner_seg_dataset(self, fix_seed, dataset, ner_tokenizer):
        essay_feedback_dataset = dataset.make_ner_dataset(tokenizer=ner_tokenizer,
                                                          seg_only=True)
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (1024,))
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (1024,))
        assert(isinstance(essay_feedback_dataset[0][2], torch.Tensor))
        assert(essay_feedback_dataset[0][2].size() == (1024,))
        assert(essay_feedback_dataset[0][2].max().item() <= 15)
        assert(essay_feedback_dataset[0][2].min().item() == -1)

    def test_make_ner_dataset(self, fix_seed, dataset, ner_tokenizer):
        essay_feedback_dataset = dataset.make_ner_dataset(tokenizer=ner_tokenizer,
                                                          seg_only=False)
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (1024,))
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (1024,))
        assert(isinstance(essay_feedback_dataset[0][2], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (1024,))

    def test_make_essay_feedback_dataset(self, fix_seed, dataset, kls_model):
        essay_feedback_dataset = dataset.make_essay_feedback_dataset(encoder=kls_model)
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (32, 769))
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (32, 1))

    def test_make_essay_feedback_dataset_random(self, fix_seed, dataset, kls_model):
        essay_feedback_dataset = dataset.make_essay_feedback_dataset(
            encoder=kls_model, randomize_segments=True)
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (32, 769))
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (32, 1))



