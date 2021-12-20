from pathlib import Path

import pandas as pd

from dataset import *

def test_init():
    dataset = ArgumentDataset()
    assert(len(dataset) == 144293)

def test_idx():
    dataset = ArgumentDataset()
    row = dataset[0:10]
    assert(len(row) == 10)

def test_essays():
    dataset = ArgumentDataset()
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

def test_arguments():
    dataset = ArgumentDataset()
    arguments = dataset.arguments(20)
    assert(len(list(arguments)) == 20)

def test_essay_paths():
    dataset = ArgumentDataset()
    paths = dataset.essay_paths()
    assert(isinstance(paths[0], Path))
    assert(len(paths) == 15594)

def test_random_essay():
    dataset = ArgumentDataset()
    random_essay = dataset.random_essay(num_essays=5)
    assert(len(random_essay) == 5)
    assert(isinstance(random_essay[0], str))

def test_random_span():
    dataset = ArgumentDataset()
    random_span = dataset.random_span(num_words=5, num_spans=10)
    assert(len(random_span) == 10)
    assert(isinstance(random_span[0], str))
    assert(len(random_span[0].split()) == 5)

