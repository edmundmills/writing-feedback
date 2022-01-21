from csv import DictReader
import random
from typing import Any, List, Tuple

import pandas as pd
import tqdm

from utils.constants import data_path, essay_dir, label_file
from core.essay import Essay
 

class EssayDataset:
    def __init__(self, n_essays=None, essay_ids=None, full_dataset=None) -> None:
        self.ner_probs = {}
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
                essay = (text, labels)
            else:
                essay = full_dataset.essays[essay_id]
            self.essays[essay_id] = essay
        if full_dataset and len(full_dataset.ner_probs) > 0:
            self.ner_probs = full_dataset.ner_probs
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
        essay_text, essay_labels = self.essays[essay_id]
        essay = Essay(essay_id, essay_text, essay_labels)
        essay.ner_pobs = self.ner_probs.get(essay_id, None)
        return essay

    def __len__(self):
        return len(self.essay_ids)

    def __getitem__(self, idx) -> Tuple[str, str, pd.DataFrame]:
        essay_id = self.essay_ids[idx]
        return self.get_by_id(essay_id)

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

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'ner_probs' and 'essay_ids' in self.__dict__:
            pruned_dict = {}
            for essay_id in self.essay_ids:
                pruned_dict[essay_id] = __value[essay_id]
            __value = pruned_dict
        object.__setattr__(self, __name, __value)
