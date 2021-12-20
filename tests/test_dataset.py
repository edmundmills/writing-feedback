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
    essays = dataset.essays()
    essay = next(essays)
    assert(isinstance(essay, pd.DataFrame))
    assert(len(essay) == 10)
    ids = list(essay.loc[:,'id'])
    assert(all(id == ids[0] for id in ids))
    next_essay = next(essays)
    next_ids = list(next_essay.loc[:,'id'])
    assert(ids[0] != next_ids[0])
    essay_count = sum(1 for _ in dataset.essays())
    assert(essay_count == 15594)

