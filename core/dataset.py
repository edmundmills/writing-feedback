from itertools import permutations
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

argument_types = {
            'Lead': 0,
            'Position': 1,
            'Claim': 2,
            'Counterclaim': 3,
            'Rebuttal': 4,
            'Evidence': 5,
            'Concluding Statement': 6
        }

def df_train_val_split(df, train_val_split):
    msk = np.random.randn(len(df)) < train_val_split
    return df[msk], df[~msk]

def df_to_text_and_label(df):
    text = list(df.loc[:,'discourse_text'])
    labels = torch.LongTensor(list(argument_types[row.loc['discourse_type']] for _, row in df.iterrows()))
    return text, labels
 
class ClassificationDataset:
    def __init__(self, text:List[str], labels:torch.Tensor):
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx, ...]

    def __len__(self):
        return len(self.text)


class ComparisonDataset:
    def __init__(self, text_pairs:List[Tuple[str]], labels:torch.Tensor):
        self.text_pairs = text_pairs
        self.labels = labels

    def __getitem__(self, idx):
        return self.text_pairs[idx], self.labels[idx, ...]

    def __len__(self):
        return len(self.text_pairs)


class ArgumentDataset:
    def __init__(self, nrows=None) -> None:
        self.data_path = Path('data')
        self.essay_dir = self.data_path / 'train'
        self.df = pd.read_csv(self.data_path / 'train.csv', nrows=nrows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> pd.DataFrame:
        return self.df.iloc[idx]

    @property
    def essay_paths(self) -> List[Path]:
        return list(self.essay_dir.iterdir())

    def essays(self, number=None):
        idx = 0
        count = 0
        number = number or float('Inf')
        while idx < len(self) and count < number:
            count += 1
            init_idx = idx
            essay_id = self[idx].loc['id']
            while idx < len(self) and self[idx].loc['id'] == essay_id:
                idx += 1
            yield self[init_idx:idx]

    def labels_by_id(self, id):
        matches = self.df.loc[:,'id'] == id
        return self.df[matches]

    def arguments(self, number=None, arg_type=None):
        number = number or float('Inf')
        if arg_type:
            indices = self.df['discourse_type'] == arg_type
            df = self.df[indices]
        else:
            df = self.df
        return df.iloc[:number]

    def open_essay(self, essay_id):
        path = self.data_path / 'train' / f'{essay_id}.txt'
        with open(path) as f:
            essay_text = f.read()
            essay_labels = self.labels_by_id(essay_id)
        return essay_text, essay_labels

    def random_essay(self, num_essays=1, essay_list=None) -> List[str]:
        essay_paths = essay_list or self.essay_paths
        essays = []
        for essay_path in random.choices(essay_paths, k=num_essays):
            essay_id = essay_path.stem
            essays.append((essay_id, essay_path, *self.open_essay(essay_id)))
        return essays

    def random_argument(self, num_args=1, essay_list=None):
        essays = self.random_essay(num_essays=num_args, essay_list=essay_list)
        random_args = [random.choices(list(labels.loc[:,'discourse_text']), k=1)[0]
                       for _, _, _, labels in essays]
        return random_args

    def random_span(self, num_words, num_spans=1):
        spans = []
        for _, _, essay_text, _ in self.random_essay(num_spans):
            words = essay_text.split()
            start_idx = len(words) - num_words
            if start_idx < 0:
                spans.append(essay_text)
            else:
                start_idx = random.sample(range(start_idx), 1)[0]
                stop_idx = start_idx + num_words
                words = words[start_idx:stop_idx]
                span = ' '.join(words)
                spans.append(span)
        return spans

    def make_arg_classification_datasets(self, train_val_split=0.9) -> Tuple[ClassificationDataset, ClassificationDataset]:
        print('Making Argmument Classification Dataset')
        train_df, val_df = df_train_val_split(self.df, train_val_split)
        train_text, train_labels = df_to_text_and_label(train_df)
        val_text, val_labels = df_to_text_and_label(val_df)
        print(f'Argument Classification Dataset Created with {len(train_text)} training samples and {len(val_text)} validation samples')
        return ClassificationDataset(train_text, train_labels), ClassificationDataset(val_text, val_labels)

    def make_polarity_dataset(self, train_val_split=0.9, n_essays=None):
        n_essays = n_essays or len(self.essay_paths)
        print(f'Making polarity dataset for {n_essays} essays')
        n_train_essays = int(n_essays * train_val_split)
        essays = self.essay_paths.copy()
        essays = essays[:n_essays]
        random.shuffle(essays)
        train_text_pairs = []
        train_labels = []
        val_text_pairs = []
        val_labels = []
        for idx, essay in enumerate(essays):
            essay_id = essay.stem
            if idx <= n_train_essays:
                pairs, labels = self.polarity_pairs(essay_id, essays[:n_train_essays])
                train_text_pairs.extend(pairs)
                train_labels.extend(labels)
            else:
                pairs, labels = self.polarity_pairs(essay_id, essays[n_train_essays:])
                val_text_pairs.extend(pairs)
                val_labels.extend(labels)
        train_labels = torch.FloatTensor(train_labels)
        val_labels = torch.FloatTensor(val_labels)
        print(f'Polarity Dataset Created with {len(train_text_pairs)} training pairs and {len(val_text_pairs)} validation pairs')
        return ComparisonDataset(train_text_pairs, train_labels), ComparisonDataset(val_text_pairs, val_labels)

    def polarity_pairs(self, essay_id, reference_essay_list=None):
        _, essay_labels = self.open_essay(essay_id)
        text_pairs = []
        labels = []
        position = essay_labels[essay_labels['discourse_type']=='Position']
        position_text = position.iloc[0]['discourse_text'] if len(position) > 0 else None
        conclusion = essay_labels[essay_labels['discourse_type']=='Concluding Statement']
        conclusion_text = conclusion.iloc[0]['discourse_text']  if len(conclusion) > 0 else None
        claims = list(essay_labels[essay_labels['discourse_type']=='Claim']['discourse_text'])
        counterclaims = list(essay_labels[essay_labels['discourse_type']=='Counterclaim']['discourse_text'])
        evidence = list(essay_labels[essay_labels['discourse_type']=='Evidence']['discourse_text'])
        prev_row = None
        for _, row in essay_labels.iterrows():
            if prev_row is not None:
                if prev_row['discourse_type'] == 'Claim' and row['discourse_type'] == 'Evidence':
                    text_pairs.append((prev_row['discourse_text'], row['discourse_text']))
                    labels.append(1)
                elif prev_row['discourse_type'] == 'Claim' and row['discourse_type'] == 'Counterclaim':
                    text_pairs.append((prev_row['discourse_text'], row['discourse_text']))
                    labels.append(-1)
                elif prev_row['discourse_type'] == 'Counterclaim' and row['discourse_type'] == 'Rebuttal':
                    text_pairs.append((prev_row['discourse_text'], row['discourse_text']))
                    labels.append(-1)
            prev_row = row
        if position_text:
            text_pairs.extend(((position_text, claim) for claim in claims))
            labels.extend(1 for _ in claims)
            text_pairs.extend(((position_text, claim) for claim in counterclaims))
            labels.extend(-1 for _ in counterclaims)
        if conclusion_text:
            text_pairs.extend(((conclusion_text, claim) for claim in claims))
            labels.extend(1 for _ in claims)
            text_pairs.extend(((conclusion_text, claim) for claim in counterclaims))
            labels.extend(-1 for _ in counterclaims)
        if evidence and len(evidence) >= 2:
            text_pairs.extend(permutations(evidence, 2))
            labels.extend(0 for _ in permutations(evidence, 2))
        reference_essay_list = reference_essay_list or self.essay_paths
        random_args = self.random_argument(num_args=int(len(text_pairs)/2), essay_list=reference_essay_list)
        essay_args = list(essay_labels.loc[:,'discourse_text'])
        rand_essay_args = random.choices(essay_args, k=len(random_args))
        rand_first = random.choices((True, False), k=len(random_args))
        for rand_first, random_arg, rand_essay_arg in zip(rand_first, random_args, rand_essay_args):
            if rand_first:
                pair = (random_arg, rand_essay_arg)
            else:
                pair = (rand_essay_arg, random_arg)
            text_pairs.append(pair)
            labels.append(0)
        return text_pairs, labels
