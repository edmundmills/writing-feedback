from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset import *

def test_df_train_val_split(fix_seed):
    df = pd.DataFrame(np.random.randn(100, 2))
    train, val = df_train_val_split(df, 0.8)
    assert(isinstance(train, pd.DataFrame))
    assert(isinstance(val, pd.DataFrame))
    assert(len(train) == 86)
    assert(len(val) == 14)

def test_df_to_text_and_label(dataset):
    df = dataset[:10]
    text, label = df_to_text_and_label(df)
    assert(len(text) == 10)
    assert(isinstance(text, list))
    assert(isinstance(text[0], str))
    assert(isinstance(label, torch.Tensor))

def test_init(dataset):
    assert(len(dataset) == 100)

def test_idx(dataset):
    row = dataset[0:10]
    assert(len(row) == 10)

def test_essays(dataset):
    essays = dataset.essays(10)
    essay = next(essays)
    assert(isinstance(essay, pd.DataFrame))
    assert(len(essay) == 10)
    ids = list(essay.loc[:,'id'])
    assert(all(id == ids[0] for id in ids))
    next_essay = next(essays)
    next_ids = list(next_essay.loc[:,'id'])
    assert(ids[0] != next_ids[0])
    essay_count = sum(1 for _ in dataset.essays(10))
    assert(essay_count == 10)

def test_arguments(dataset):
    arguments = dataset.arguments(20)
    assert(len(arguments) == 20)
    assert(sum(arguments.loc[:,'discourse_type'] == 'Claim') < 20)
    arguments = dataset.arguments(20, arg_type='Claim')
    assert(len(arguments) == 20)
    assert(sum(arguments.loc[:,'discourse_type'] == 'Claim') == 20)

def test_essay_paths(dataset):
    paths = dataset.essay_paths
    assert(isinstance(paths[0], Path))
    assert(len(paths) == 15594)

def test_random_essay(dataset):
    random_essay = dataset.random_essay(num_essays=5)
    assert(len(random_essay) == 5)
    assert(isinstance(random_essay[0], tuple))
    assert(isinstance(random_essay[0][0], str))
    assert(isinstance(random_essay[0][1], Path))
    assert(isinstance(random_essay[0][2], str))
    assert(isinstance(random_essay[0][3], pd.DataFrame))

def test_random_span(dataset):
    random_span = dataset.random_span(num_words=5, num_spans=10)
    assert(len(random_span) == 10)
    assert(isinstance(random_span[0], str))
    assert(len(random_span[0].split()) == 5)

def test_labels_by_id(dataset):
    essays = dataset.essays(1)
    essay = next(essays)
    essay_id = essay.iloc[0].loc['id']
    lookup_essay = dataset.labels_by_id(essay_id)
    assert(essay.equals(lookup_essay))

def test_make_arg_class_dataset(fix_seed, dataset):
    train, val = dataset.make_arg_classification_datasets()
    assert(isinstance(train, ClassificationDataset))
    assert(isinstance(val, ClassificationDataset))
    assert(len(train) + len(val) == len(dataset))
    assert(len(train) == 78)
    assert(isinstance(train[0][0], str))
    assert(isinstance(train[0][1], torch.Tensor))
    assert(len(train) == len(train.text))
    assert(len(train) == train.labels.size()[0])
    assert(len(val) == len(val.text))
    assert(len(val) == val.labels.size()[0])

def test_open_essay(dataset):
    essay_text, labels = dataset.open_essay('423A1CA112E2') 
    assert(isinstance(essay_text, str))
    assert(isinstance(labels, pd.DataFrame))
    assert(len(essay_text) > 0)
    assert(len(labels) > 0)

def test_polarity_pairs():
    dataset = ArgumentDataset()
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

def test_make_polarity_dataset():
    dataset = ArgumentDataset()
    train, val = dataset.make_polarity_dataset(n_essays=20)
    assert(isinstance(train, ComparisonDataset))
    assert(isinstance(val, ComparisonDataset))
    assert(len(train) == len(train.text_pairs))
    assert(len(train) == train.labels.size()[0])
    assert(len(val) == len(val.text_pairs))
    assert(len(val) == val.labels.size()[0])
    assert(isinstance(val[0], tuple))
    assert(all(len(item) == 2 for item in val[:][0]))
    assert(all(type(item[0]) == str for item in val[:][0]))
    assert(isinstance(val[0][0], tuple))
    assert(isinstance(val[0][1], torch.Tensor))
    assert(isinstance(val[0][0][0], str))
    assert(isinstance(val[0][0][1], str))
