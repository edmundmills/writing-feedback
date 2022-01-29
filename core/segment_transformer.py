from collections import defaultdict
import itertools
import random

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import torch
import tqdm

from core.dataset import SegmentTokens

from utils.networks import Model
from utils.constants import de_type_to_num, de_num_to_type

class SegmentTransformer(Model):
    def __init__(self, seg_t_args) -> None:
        super().__init__()
        print('Loading Sentence Transformer...')
        self.model = SentenceTransformer('all-distilroberta-v1')
        self.train_on_joins = seg_t_args.train_on_joins
        self.train_on_classes = seg_t_args.train_on_classes
        self.train_on_polarity = seg_t_args.train_on_polarity
        print('Sentence Transformer Loaded')

    def make_join_dataset(self, essay_dataset):
        print('Making Join Dataset...')
        join_examples = []
        for essay in tqdm.tqdm(essay_dataset):
            _, essay_lens, labels = essay.segments
            segment_text = essay.text_from_segments(essay_lens, join=True)
            for t1, t2, l1, l2 in zip(segment_text[:-1], segment_text[1:],
                                      labels[:-1], labels[1:]):
                if l2 > 7 or ((l1 == 0) ^ (l2 == 0)):
                    join_examples.append(
                        InputExample(texts=[t1, t2], label=1.0)
                    )
                elif l1 != 0 and l2 != 0:
                    join_examples.append(
                        InputExample(texts=[t1, t2], label=0.0)
                    )
        print(f'Join Dataset created with {len(join_examples)} samples')
        print(f'{sum(example.label for example in join_examples)/len(join_examples):.2f} of samples are to join')
        return join_examples

    def make_classification_dataset(self, essay_dataset, num_samples=100000):
        de_types = defaultdict(list)
        print('Making Classification Dataset...')
        for essay in tqdm.tqdm(essay_dataset):
            _, essay_lens, labels = essay.segments
            segment_text = essay.text_from_segments(essay_lens, join=True)
            labels = labels - (7 * (labels > 7))
            for de_type in de_num_to_type:
                de_types[de_type].extend([segment_text for segment_text, label
                                          in zip(segment_text, labels)
                                          if label == de_type_to_num[de_type]])
        examples = []
        num_same = min(num_samples, min(len(segs) for segs in de_types.values()))
        num_diff = (num_same // 7) + 1
        for de_type in de_num_to_type:
            for other_de_type in de_num_to_type:
                if de_type == other_de_type:
                    combs = itertools.combinations(de_types[de_type], 2)
                    examples.extend(
                        [InputExample(texts=[t1, t2], label=1.0)
                         for t1, t2 in itertools.islice(combs, num_same)]
                    )
                else:
                    segs = np.random.choice(range(len(de_types[de_type])),
                                            size=num_diff, replace=False)
                    others = np.random.choice(range(len(de_types[other_de_type])),
                                              size=num_diff, replace=False)
                    examples.extend(
                        [InputExample(texts=[de_types[de_type][i1], de_types[other_de_type][i2]],
                                      label=0.0)
                         for i1, i2 in zip(segs, others)]
                    )
        print(f'Classification Dataset Created with {len(examples)} Samples')
        print(f'{sum(example.label for example in examples)/len(examples):.2f} of samples are matches')
        return examples


    def make_polarity_dataset(self, essay_dataset):
        print('Making Polarity Dataset...')
        examples = []
        for essay in tqdm.tqdm(essay_dataset):
            _, essay_lens, labels = essay.segments
            segment_text = essay.text_from_segments(essay_lens, join=True)
            labels = np.array(labels)
            labels = labels - (7 * (labels > 7))
            de_types = {}
            for de_type in de_num_to_type:
                de_types[de_type] = [idx for idx, label in enumerate(labels)
                                     if label == de_type_to_num[de_type]]
            
            positives = de_types['Position'] + de_types['Claim'] + de_types['Rebuttal'] \
                    + de_types['Concluding Statement']
            contra = []
            for cc in de_types['Counterclaim']:                
                contra.extend([(cc, other) for other in positives])
                contra.extend([(other, cc) for other in positives])
            support = list(itertools.permutations(positives, 2))
            if len(support) < len(contra):
                contra_choices = np.random.choice(range(len(contra)), size=len(support), replace=False)
                contra = [contra[idx] for idx in contra_choices]
            else:
                support_choices = np.random.choice(range(len(support)), size=len(contra), replace=False)
                support = [support[idx] for idx in support_choices]
            examples.extend(
                [InputExample(texts=[segment_text[i1], segment_text[i2]],
                             label=0.0) for i1, i2 in contra]
            )
            examples.extend(
                [InputExample(texts=[segment_text[i1], segment_text[i2]],
                             label=1.0) for i1, i2 in support]
            )
        print(f'Polarity Dataset created with {len(examples)} samples')
        print(f'{1 - (sum(example.label for example in examples)/(1+len(examples))):.2f} of pairs are opposed')
        return examples


    def finetune(self, train_dataset, val_dataset, seg_t_args):
        join_examples_train = self.make_join_dataset(train_dataset)
        join_examples_val = self.make_join_dataset(val_dataset)
        polarity_examples_train = self.make_polarity_dataset(train_dataset)
        polarity_examples_val = self.make_polarity_dataset(val_dataset)
        classification_examples_train = self.make_classification_dataset(train_dataset)
        classification_examples_val = self.make_classification_dataset(val_dataset)
        join_dataloader = torch.utils.data.DataLoader(join_examples_train,
                                                       shuffle=True,
                                                       batch_size=seg_t_args.batch_size)
        polarity_dataloader = torch.utils.data.DataLoader(polarity_examples_train,
                                                          shuffle=True,
                                                          batch_size=seg_t_args.batch_size)
        classification_dataloader = torch.utils.data.DataLoader(classification_examples_train,
                                                                shuffle=True,
                                                                batch_size=seg_t_args.batch_size)
        val_examples = join_examples_val + polarity_examples_val + classification_examples_val
        if len(val_examples) > seg_t_args.eval_samples:
            val_examples = random.sample(val_examples, seg_t_args.eval_samples)
        print(f'Evaluating with {len(val_examples)} samples')
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples)      

        loss_func = losses.ContrastiveLoss(self.model)
        self.model.fit([(join_dataloader, loss_func),
                        (polarity_dataloader, loss_func),
                        (classification_dataloader, loss_func)],
                        evaluator=evaluator,
                        evaluation_steps=100,
                        warmup_steps=100,
                        steps_per_epoch=seg_t_args.steps_per_epoch,
                        output_path='sentence_transformers')
        print('Finetuning Complete')
    
    def encode(self, segments, max_d_elems):
        num_segments = len(segments)
        encoded = self.model.encode(segments, convert_to_numpy=False, convert_to_tensor=True).cpu()
        attention_mask = torch.ones_like(encoded)
        encoded = torch.cat((encoded[:max_d_elems,:],
                                    torch.ones(max_d_elems - num_segments, 768)))
        attention_mask = torch.cat((attention_mask[:max_d_elems,:],
                                    torch.zeros(max_d_elems - num_segments, 768)))
        return encoded, attention_mask[...,0].bool()

    def tokenize_dataset(self, essay_dataset, max_d_elems=40):
        print('Tokenizing Dataset Segments...')
        for essay in tqdm.tqdm(essay_dataset):
            _essay_ner_features, seg_lens, essay_labels = essay.segments
            text = essay.text_from_segments(seg_lens, join=True)
            encoded, attention_mask = self.encode(text, max_d_elems)
            segment_tokens = SegmentTokens(encoded, attention_mask)
            essay_dataset.essays[essay.essay_id]['segment_tokens'] = segment_tokens
        print(f'Dataset Tokenized')
        return essay_dataset
