import random
from pathlib import Path
from typing import List

import pandas as pd

class ArgumentDataset:
    def __init__(self) -> None:
        self.data_path = Path('data')
        self.essay_dir = self.data_path / 'train'
        self.df = pd.read_csv(self.data_path / 'train.csv')
        self.argument_types = (
            'Lead',
            'Position',
            'Claim',
            'Counterclaim',
            'Rebuttal',
            'Evidence',
            'Concluding Statement'
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> pd.DataFrame:
        return self.df.iloc[idx]

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

    def random_essay(self, num_essays=1) -> List[str]:
        essays = []
        for essay_path in random.sample(self.essay_paths(), num_essays):
            with open(essay_path) as f:
                essay_id = essay_path.stem
                essay_text = f.read()
                essay_labels = self.labels_by_id(essay_id)
                essays.append((essay_id,
                               essay_path,
                               essay_text,
                               essay_labels))
        return essays

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
