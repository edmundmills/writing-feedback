from csv import DictReader
from collections import defaultdict
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import tqdm

from core.constants import data_path, essay_dir, label_file, argument_names, argument_types
from core.essay import Essay
from utils.grading import get_discourse_elements, get_labels, to_tokens
 
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
                if len(full_dataset.ner_probs) > 0:
                    self.ner_probs[essay_id] = full_dataset.ner_probs[essay_id]
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
        essay_text, essay_labels = self.essays[essay_id]
        return Essay(essay_id, essay_text, essay_labels)

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

    def make_polarity_dataset(self) -> ComparisonDataset:
        text_pairs = []
        labels = []
        for essay in tqdm.tqdm(self):
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

    def make_ner_dataset(self, tokenizer, seg_only=False) -> TensorDataset:
        print('Making NER Dataset')
        input_ids = []
        attention_masks = []
        labels = []
        all_word_ids = []
        for essay in tqdm.tqdm(self):
            encoded = tokenizer.encode(essay.text)
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            word_ids = encoded['word_ids']
            label_tokens = to_tokens(essay.correct_predictions, tokenizer.max_tokens)
            def get_label(word_idx, label_tokens):
                if word_idx is None:
                    return [-1,-1]
                else:
                    return label_tokens[:,word_idx]
            label_tokens = np.array([get_label(word_idx, label_tokens) for word_idx in word_ids]).T
            if seg_only:
                label_tokens = label_tokens[0]
                label_tokens = label_tokens[np.newaxis, ...]
            labels.append(torch.LongTensor(label_tokens).squeeze())
            all_word_ids.append(torch.LongTensor(word_ids).squeeze())
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)
        all_word_ids = torch.stack(all_word_ids, dim=0)
        dataset = TensorDataset(input_ids, attention_masks, labels, all_word_ids)
        print('NER Dataset Created')
        return dataset

    def get_ner_probs(self, tokenizer, ner_model) -> None:
        print('Getting NER Probs')
        self.ner_probs = {}
        for essay in tqdm.tqdm(self):
            encoded = tokenizer.encode(essay.text)
            probs = ner_model.inference(encoded['input_ids'],
                                        encoded['attention_mask'],
                                        encoded['word_id_tensor'])
            self.ner_probs[essay.essay_id] = probs
        print('NER Probs Added to Dataset')
        return self.ner_probs

    def make_essay_feedback_dataset(self, encoder, randomize_segments=False) -> TensorDataset:
        print('Making Essay Feedback Dataset')
        encoded_text = []
        labels = []
        for essay in tqdm.tqdm(self):
            if not randomize_segments:
                essay_encoded_text = encoder.encode(essay.d_elems_text)
                token_len = essay_encoded_text.size(0)
                essay_labels = [argument_types[text_label] for text_label
                                in essay.labels.loc[:,'discourse_type'].tolist()]
                essay_labels = essay_labels[:token_len] + [-1]*max(0, token_len - len(essay_labels))
            else:
                pstrings = essay.random_pstrings(max_d_elems=encoder.max_d_elems)
                essay_labels = get_labels(pstrings, essay, num_d_elems=encoder.max_d_elems)
                d_elems = get_discourse_elements(essay.text, pstrings)
                essay_encoded_text = encoder.encode(d_elems)
            encoded_text.append(essay_encoded_text)
            labels.append(essay_labels)
        text_tensor = torch.stack(encoded_text, dim=0)
        label_tensor = torch.LongTensor(labels).unsqueeze(-1)
        dataset = TensorDataset(text_tensor, label_tensor)
        print('Essay Feedback Dataset Created')
        return dataset

    def make_bc_dataset(self, seqwise_env) -> TensorDataset:
        states = defaultdict(list)
        actions = []
        for essay in tqdm.tqdm(self):
            state = seqwise_env.reset(essay_id=essay.essay_id)
            for pred in essay.correct_predictions:
                action = len(pred)
                actions.append(action)
                next_state, reward, done, info = seqwise_env.step(action)
                for k, v in state.items():
                    states[k].append(v)
                state = next_state

        dataset = TensorDataset(
            torch.LongTensor(np.array(states['seg_tokens'])),
            torch.LongTensor(np.array(states['class_tokens'])),
            torch.FloatTensor(np.array(states['ner_probs'])),
            torch.FloatTensor(actions)
        )
        return dataset            
