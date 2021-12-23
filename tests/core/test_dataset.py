from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset import *

def test_init(dataset):
    assert(len(dataset) == 10)

def test_idx(dataset):
    essay = dataset[0]
    assert(isinstance(essay, tuple))
    assert(isinstance(essay[0], str))
    assert(isinstance(essay[1], str))
    assert(isinstance(essay[2], pd.DataFrame))

def test_random_essay(dataset):
    random_essay = dataset.random_essay(num_essays=5)
    assert(len(random_essay) == 5)
    assert(isinstance(random_essay[0], tuple))
    assert(isinstance(random_essay[0][0], str))
    assert(isinstance(random_essay[0][1], str))
    assert(isinstance(random_essay[0][2], pd.DataFrame))

def test_random_span(dataset):
    random_span = dataset.random_span(num_words=5, num_spans=10)
    assert(len(random_span) == 10)
    assert(isinstance(random_span[0], str))
    assert(len(random_span[0].split()) == 5)

def test_labels_by_id(dataset):
    essay_id, _, labels = dataset[0]
    lookup_essay = labels_by_id(dataset.df, essay_id)
    assert(labels.equals(lookup_essay))

def test_make_arg_class_dataset(fix_seed, dataset):
    class_dataset = dataset.make_arg_classification_dataset()
    assert(isinstance(class_dataset, ClassificationDataset))
    assert(isinstance(class_dataset[0][0], str))
    assert(isinstance(class_dataset[0][1], torch.Tensor))
    assert(len(class_dataset) == len(class_dataset.text))
    assert(len(class_dataset) == class_dataset.labels.size()[0])

def test_open_essay():
    essay_text = open_essay('423A1CA112E2') 
    assert(isinstance(essay_text, str))
    assert(len(essay_text) > 0)

def test_polarity_pairs(dataset):
    pairs, labels = dataset.polarity_pairs('6B4F7A0165B9')
    assert(isinstance(pairs, list))
    assert(isinstance(labels, list))
    assert(len(pairs) == len(labels))
    assert(min(labels) >= -1)
    assert(max(labels) <= 1)
    for pair, label in zip(pairs, labels):
        print(pair[0])
        print(pair[1])
        print(label)
        print('---')

def test_split(dataset):
    train, val = dataset.split()
    assert(isinstance(train, EssayDataset))
    assert(isinstance(val, EssayDataset))
    assert(len(train) == 9)
    assert(len(val) == 1)
    assert(set(train.df.loc[:,'id'].unique()) == set(train.essay_ids))
    assert(set(val.df.loc[:,'id'].unique()) == set(val.essay_ids))

def test_make_polarity_dataset(dataset):
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
