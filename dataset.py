from pathlib import Path

import pandas as pd

class ArgumentDataset:
    def __init__(self):
        self.data_path = Path('data')
        self.df = pd.read_csv(self.data_path / 'train.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def essays(self):
        idx = 0
        while idx < len(self):
            init_idx = idx
            essay_number = self[idx].loc['id']
            while idx < len(self) and self[idx].loc['id'] == essay_number:
                idx += 1
            yield self[init_idx:idx]

