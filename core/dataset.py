from csv import DictReader
from itertools import permutations
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import tqdm

argument_types = {
            'None': 0,
            'Lead': 1,
            'Position': 2,
            'Claim': 3,
            'Counterclaim': 4,
            'Rebuttal': 5,
            'Evidence': 6,
            'Concluding Statement': 7
        }
def get_value(item):
    return item[1]
argument_names = [k for k, _ in sorted(argument_types.items(), key=get_value)]

def load_n_essays(n_essays):
    data_path = Path('data') / 'train.csv'
    essay_ids = set()
    essay_ids_tmp = set()
    with open(data_path, 'r') as f:
        for idx, row in enumerate(DictReader(f)):
            essay_ids_tmp.add(row['id'])
            if len(essay_ids_tmp) > n_essays:
                nrows = idx
                break
            else:
                essay_ids.add(row['id'])
    df = pd.read_csv(data_path, nrows=nrows)
    return list(essay_ids), df

def labels_by_id(df, id):
    matches = df.loc[:,'id'] == id
    return df[matches]

def open_essay(essay_id):
    path = Path('data') / 'train' / f'{essay_id}.txt'
    with open(path) as f:
        essay_text = f.read()
    return essay_text

 
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


class EssayDataset:
    def __init__(self, n_essays=None, essay_ids=None, full_dataset=None) -> None:
        self.data_path = Path('data')
        self.essay_dir = self.data_path / 'train'
        if n_essays:
            print(f'Loading data for {n_essays} essays')
            essay_ids, self.df = load_n_essays(n_essays=n_essays)
        elif essay_ids:
            print(f'Loading data for {len(essay_ids)} essays')
            self.df = pd.read_csv(self.data_path / 'train.csv')
            self.df = self.df[self.df['id'].isin(essay_ids)]
        else:
            print('Loading essays...')
            self.df = pd.read_csv(self.data_path / 'train.csv')
        self.essay_ids = essay_ids or list(essay_path.stem for essay_path in self.essay_dir.iterdir())
        self.essays = {}
        if full_dataset is None:
            print('Collating essays and labels...')
        for essay_id in tqdm.tqdm(self.essay_ids):
            if full_dataset is None:
                text = open_essay(essay_id)
                labels = labels_by_id(self.df, essay_id)
            else:
                text, labels = full_dataset.essays[essay_id]
            self.essays[essay_id] = (text, labels)
        print(f'Essay dataset created with {len(self)} essays.')

    def __len__(self):
        return len(self.essay_ids)

    def __getitem__(self, idx) -> Tuple[str, str, pd.DataFrame]:
        essay_id = self.essay_ids[idx]
        essay_text, essay_labels = self.essays[essay_id]
        return essay_id, essay_text, essay_labels

    def random_essay(self, num_essays=1) -> List:
        return random.choices(self, k=num_essays)

    def random_span(self, num_words, num_spans=1):
        spans = []
        for _, essay_text, _ in self.random_essay(num_spans):
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

    def split(self, fraction=0.9):
        print('Splitting Essay Dataset')
        essays = self.essay_ids.copy()
        random.shuffle(essays)
        n_train = int(fraction * len(self))
        train_essays = self.essay_ids[:n_train]
        val_essays = self.essay_ids[n_train:]
        print(f'{len(train_essays)} Train Essays and {len(val_essays)} Validation Essays')
        return EssayDataset(essay_ids=train_essays, full_dataset=self), EssayDataset(essay_ids=val_essays, full_dataset=self)

    def make_polarity_dataset(self) -> ComparisonDataset:
        text_pairs = []
        labels = []
        for essay_id in self.essay_ids:
            pairs, polarity_labels = self.polarity_pairs(essay_id)
            text_pairs.extend(pairs)
            labels.extend(polarity_labels)
        labels = torch.FloatTensor(labels)
        print(f'Polarity Dataset Created with {len(text_pairs)} pairs.')
        return ComparisonDataset(text_pairs, labels)

    def make_arg_classification_dataset(self) -> ClassificationDataset:
        print('Making Argmument Classification Dataset')
        text = list(self.df.loc[:,'discourse_text'])
        labels = torch.LongTensor(list(argument_types[row.loc['discourse_type']] for _, row in self.df.iterrows()))
        print(f'Argument Classification Dataset Created with {len(text)} samples.')
        return ClassificationDataset(text, labels)

    def polarity_pairs(self, essay_id):
        _, essay_labels = self.essays[essay_id]
        text_pairs = []
        labels = []
        essay_arguments = essay_labels[['discourse_type', 'discourse_text']].values.tolist()
        lead = None
        position = None
        conclusion = None
        claims = []
        counterclaims = []
        evidences = []
        prev_arg = None
        for arg_type, arg_text in essay_arguments:
            if arg_type == 'Lead':
                lead = arg_text
            elif arg_type == 'Position':
                position = arg_text
            elif arg_type == 'Concluding Statement':
                conclusion = arg_text
            elif arg_type == 'Claim':
                claims.append(arg_text)
            elif arg_type == 'Counterclaim':
                counterclaims.append(arg_text)
            elif arg_type == 'Evidence':
                evidences.append(arg_text)
            if prev_arg is not None:
                if prev_arg[0] == 'Claim' and arg_type == 'Evidence':
                    text_pairs.append((prev_arg[1], arg_text))
                    labels.append(1)
                elif prev_arg[0] == 'Claim' and arg_type == 'Counterclaim':
                    text_pairs.append((prev_arg[1], arg_text))
                    labels.append(-1)
                elif prev_arg[0] == 'Counterclaim' and arg_type == 'Rebuttal':
                    text_pairs.append((prev_arg[1], arg_text))
                    labels.append(-1)
            prev_arg = arg_type, arg_text
        if position:
            text_pairs.extend(((position, claim) for claim in claims))
            labels.extend(1 for _ in claims)
            text_pairs.extend(((position, claim) for claim in counterclaims))
            labels.extend(-1 for _ in counterclaims)
        if conclusion:
            text_pairs.extend(((conclusion, claim) for claim in claims))
            labels.extend(1 for _ in claims)
            text_pairs.extend(((conclusion, claim) for claim in counterclaims))
            labels.extend(-1 for _ in counterclaims)
            text_pairs.extend(((evidence, conclusion) for evidence in evidences))
            labels.extend(0 for _ in evidences)
        if evidences and len(evidences) >= 2:
            text_pairs.extend(permutations(evidences, 2))
            labels.extend(0 for _ in permutations(evidences, 2))
        if lead:
            text_pairs.extend(((evidence, lead) for evidence in evidences))
            labels.extend(0 for _ in evidences)
        return text_pairs, labels
