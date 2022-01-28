from csv import DictReader
import random
from typing import Any, List, Tuple

import pandas as pd
import pickle
import tqdm

from utils.constants import data_path, essay_dir, label_file
from core.essay import Essay
 

class EssayDataset:
    def __init__(self, n_essays=None, essay_ids=None, full_dataset=None) -> None:
        if n_essays:
            print(f'Loading data for {n_essays} essays')
            essay_ids, self.df = self._load_n_essays(n_essays=n_essays)
        elif essay_ids:
            print(f'Loading data for {len(essay_ids)} essays')
            self.df = pd.read_csv(data_path / 'train.csv')
            self.df = self.df[self.df['id'].isin(essay_ids)]
        else:
            print('Loading essays...')
            self.df = pd.read_csv(data_path / 'train.csv')
        self.essay_ids = essay_ids or list(essay_path.stem for essay_path in essay_dir.iterdir())
        self.essays = {}
        if full_dataset is None:
            print('Collating essays and labels...')
        for essay_id in tqdm.tqdm(self.essay_ids):
            if full_dataset is None:
                essay_file = essay_dir / f'{essay_id}.txt'
                with open(essay_file) as f:
                    text = f.read()
                matches = self.df.loc[:,'id'] == essay_id
                labels = self.df[matches]
                essay = {'text': text,
                         'labels': labels,
                         'fold': None}
            else:
                essay = full_dataset.essays[essay_id]
            self.essays[essay_id] = essay
        print(f'Essay dataset created with {len(self)} essays.')

    def _load_n_essays(self, n_essays):
        essay_ids = set()
        essay_ids_tmp = set()
        with open(label_file, 'r') as f:
            for idx, row in enumerate(DictReader(f)):
                essay_ids_tmp.add(row['id'])
                if len(essay_ids_tmp) > n_essays:
                    nrows = idx
                    break
                else:
                    essay_ids.add(row['id'])
        df = pd.read_csv(label_file, nrows=nrows)
        return list(essay_ids), df

    def get_by_id(self, essay_id):
        essay_data = self.essays[essay_id]
        essay = Essay(essay_id=essay_id,
                      **essay_data)
        return essay

    def __len__(self):
        return len(self.essay_ids)

    def __getitem__(self, idx) -> Tuple[str, str, pd.DataFrame]:
        essay_id = self.essay_ids[idx]
        return self.get_by_id(essay_id)

    def make_folds(self, num_folds=5) -> None:
        self.folds = list(range(num_folds))
        essays = self.essay_ids.copy()
        random.shuffle(essays)
        for idx, essay_id in enumerate(essays):
            self.essays[essay_id]['fold'] = idx % num_folds
    
    def get_fold(self, fold):
        val_ids = [essay_id for essay_id in self.essay_ids if
                    self.essays[essay_id]['fold'] == fold]
        train_ids = [essay_id for essay_id in self.essay_ids if
                    self.essays[essay_id]['fold'] != fold]
        train = EssayDataset(essay_ids=train_ids, full_dataset=self)
        val = EssayDataset(essay_ids=val_ids, full_dataset=self)
        return train, val

    def random_essay(self, num_essays=1) -> List:
        return random.choices(self, k=num_essays)

    def random_span(self, num_words, num_spans=1):
        spans = []
        for essay in self.random_essay(num_spans):
            words = essay.text.split()
            start_idx = len(words) - num_words
            if start_idx < 0:
                spans.append(essay.text)
            else:
                start_idx = random.sample(range(start_idx), 1)[0]
                stop_idx = start_idx + num_words
                words = words[start_idx:stop_idx]
                span = ' '.join(words)
                spans.append(span)
        return spans

    def split(self, sections=[0.9, 0.1]):
        print('Splitting Essay Dataset')
        if len(sections) > len(self):
            raise ValueError('Cant split dataset into more sections than there are essays')
        essays = self.essay_ids.copy()
        random.shuffle(essays)
        datasets = []
        ds_sizes = [int(fraction * len(self)) for fraction in sections]
        start_idx = 0
        for idx, size in enumerate(ds_sizes):
            end_idx = start_idx + size
            if idx == len(ds_sizes) - 1:
                end_idx = len(self)
            ds_essays = self.essay_ids[start_idx:end_idx]
            datasets.append(EssayDataset(essay_ids=ds_essays, full_dataset=self))
            start_idx = end_idx
        print(f'Dataset split into datasets with sizes {[len(ds) for ds in datasets]}')
        return datasets

    def save(self, path):
        print(f'Saving Dataset to {str(path)}')
        with open(path, 'wb') as save_file:
            pickle.dump(self, save_file)
        print('Dataset Saved')

    @classmethod
    def load(cls, path):
        print(f'Loading dataset from {str(path)}')
        with open(path, 'rb') as saved_file:
            dataset = pickle.load(saved_file)
        if not isinstance(dataset, cls):
            raise TypeError('File does not contain a dataset')
        save = False
        if 'segments' in dataset.__dict__:
            for k, v in dataset.segments.items():
                dataset.essays[k]['segments'] = v
            delattr(dataset, 'segments')
            save = True
        if 'ner_probs' in dataset.__dict__:
            for k, v in dataset.ner_probs.items():
                dataset.essays[k]['ner_probs'] = v
            delattr(dataset, 'ner_probs')
            save = True
        if save:
            dataset.save(str(path))
        print(f'Dataset Loaded with {len(dataset)} essays')
        return dataset

    def __add__(self, ds):
        new_dataset = self.split([1])[0]
        new_dataset.essay_ids += ds.essay_ids
        new_dataset.essays.update(ds.essays)
        new_dataset.df = pd.concat((new_dataset.df, ds.df))
        return new_dataset
    
    def copy_essays(self, other):
        for essay_id in self.essay_ids:
            self.essays[essay_id] = other.essays[essay_id]