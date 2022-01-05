from csv import DictReader
import random
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset
import tqdm

from core.constants import data_path, essay_dir, label_file, argument_names, argument_types
from core.essay import Essay

 
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


    def __len__(self):
        return len(self.essay_ids)

    def __getitem__(self, idx) -> Tuple[str, str, pd.DataFrame]:
        essay_id = self.essay_ids[idx]
        essay_text, essay_labels = self.essays[essay_id]
        return Essay(essay_id, essay_text, essay_labels)

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
        for essay in self:
            pairs, polarity_labels = essay.polarity_pairs()
            text_pairs.extend(pairs)
            labels.extend(polarity_labels)
        labels = torch.FloatTensor(labels)
        print(f'Polarity Dataset Created with {len(text_pairs)} pairs.')
        return ComparisonDataset(text_pairs, labels)

    def make_arg_classification_dataset(self, balanced=False) -> ClassificationDataset:
        print('Making Argmument Classification Dataset')
        if balanced:
            type_dfs = []
            type_lens = []
            for arg_type in (arg_type for arg_type in argument_names if arg_type != 'None'):
                type_df = self.df[self.df['discourse_type'] == arg_type]
                type_lens.append(len(type_df))
                type_dfs.append(type_df)

            min_len = min(type_lens)
            text = []
            labels = []
            for df in type_dfs:
                text.extend(list(df.iloc[:min_len].loc[:,'discourse_text']))
                labels.extend([argument_types[row.loc['discourse_type']]
                              for _, row in df.iloc[:min_len].iterrows()])
        else:
            text = list(self.df.loc[:,'discourse_text'])
            labels = [argument_types[row.loc['discourse_type']] for _, row in self.df.iterrows()]
        labels = torch.LongTensor(labels)
        print(f'Argument Classification Dataset Created with {len(text)} samples.')
        return ClassificationDataset(text, labels)

    def make_essay_feedback_dataset(self, encoder) -> TensorDataset:
        encoded_text = []
        labels = []
        for essay in self:
            d_elems = essay.labels.loc[:,'discourse_text'].tolist()
            essay_encoded_text = encoder.encode(d_elems)
            token_len = essay_encoded_text.size()[0]
            essay_labels = [argument_types[text_label] for text_label
                            in essay.labels.loc[:,'discourse_type'].tolist()]
            essay_labels = essay_labels[:token_len] + [0]*max(0, token_len - len(essay_labels))
            encoded_text.append(essay_encoded_text)
            labels.append(essay_labels)
        text_tensor = torch.stack(encoded_text, dim=0)
        label_tensor = torch.LongTensor(labels).unsqueeze(-1)
        print(text_tensor.size(), label_tensor.size())
        return TensorDataset(text_tensor, label_tensor)
