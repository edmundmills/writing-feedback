from collections import deque
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import LongformerTokenizerFast, LongformerModel
import wandb

from utils.networks import MLP, Model, Mode
from core.constants import ner_token_names

class NERTokenizer:
    def __init__(self, args):
        super().__init__()
        self.max_tokens = args.essay_max_tokens
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096',
                                                                 add_prefix_space=True)

    def encode(self,text:str):
        tokenized = self.tokenizer.encode_plus(text.split(),
                                             max_length=self.max_tokens,
                                             padding='max_length',
                                             truncation=True,
                                             return_tensors='pt',
                                             return_attention_mask=True,
                                             is_split_into_words=True,
                                             )
        word_ids = tokenized.word_ids()
        word_id_tensor = torch.LongTensor(
            [word_id if word_id is not None else -1 for word_id in word_ids]
        ).unsqueeze(0)
        tokenized = {'input_ids': tokenized['input_ids'],
                     'attention_mask': tokenized['attention_mask'],
                     'word_ids': word_ids,
                     'word_id_tensor': word_id_tensor}
        return tokenized


class NERModel(Model):
    def __init__(self, args, feature_extractor=False) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        print('Loading NERModel')
        self.transformer = LongformerModel.from_pretrained(
            'allenai/longformer-base-4096').to(self.device)
        n_outputs = 2 if args.segmentation_only else 15
        self.classifier = MLP(768,
                              n_outputs,
                              args.linear_layers,
                              args.linear_layer_size,
                              dropout=0.1).to(self.device)
        print('NERModel Loaded')

    def split_essay_tokens(self, essay_tokens):
        essay_tokens = essay_tokens.long()
        input_ids, attention_mask, word_ids = torch.chunk(essay_tokens, 3, dim=-2)
        input_ids = input_ids.squeeze(-2)
        attention_mask = attention_mask.squeeze(-2)
        word_ids = word_ids.squeeze(-2)
        return input_ids, attention_mask, word_ids

    def collate_word_idxs(self, probs, word_ids):
        *_, n_essays, n_tokens = probs.size()
        range_tensor = torch.arange(n_tokens, device=probs.device)
        range_tensor = range_tensor.repeat(n_essays, 1)
        word_idxs = range_tensor + range_tensor - word_ids - 1
        msk = torch.le(word_idxs, 2 + torch.max(word_ids, dim=-1, keepdim=True).values)
        probs = torch.gather(probs, dim=-1, index=word_idxs*msk)*msk
        return probs

    def forward(self, input_ids, attention_mask=None):
        if self.feature_extractor:
            input_ids, attention_mask, word_ids = self.split_essay_tokens(input_ids)
            with torch.no_grad():
                self.eval()
                encoded = self.transformer(input_ids, attention_mask=attention_mask)
                output = self.classifier(encoded.last_hidden_state)
            output = F.softmax(output, dim=-1)
            _, output = torch.chunk(output, 2, dim=-1)
            output = output.squeeze(-1)
            output = output * attention_mask - (1 - attention_mask)
            output = self.collate_word_idxs(output, word_ids)
        else:
            encoded = self.transformer(input_ids, attention_mask=attention_mask)
            output = self.classifier(encoded.last_hidden_state)
        return output

    def train_ner(self, train_dataset, val_dataset, args):
        dataloader = DataLoader(train_dataset,
                                batch_size=args.ner.batch_size,
                                num_workers=4,
                                sampler=RandomSampler(train_dataset))

        with Mode(self, 'train'):
            step = 0
            running_loss = deque(maxlen=args.ner.print_interval)
            timestamps = deque(maxlen=args.ner.print_interval)

            for epoch in range(1, args.ner.epochs + 1):
                lr_idx = epoch - 1 if epoch <= len(args.ner.learning_rates) else - 1
                lr = args.ner.learning_rates[lr_idx]
                optimizer = torch.optim.AdamW([
                    {'params': self.transformer.parameters(), 'lr':lr},
                    {'params': self.classifier.parameters(), 'lr':lr*10}
                    ])
                print(f'Starting Epoch {epoch} with LR={lr}')
                for input_ids, attention_masks, labels in dataloader:
                    step += 1
                    input_ids = input_ids.to(self.device)
                    attention_masks = attention_masks.to(self.device)
                    labels = labels.to(self.device).squeeze()

                    logits = self(input_ids, attention_masks).squeeze()                    
                    msk = (labels != -1)
                    logits = logits[msk]
                    labels = labels[msk]
                    loss = F.cross_entropy(logits, labels)

                     
                    if step % args.ner.grad_accumulation == 0:
                        grad_steps = step // args.ner.grad_accumulation
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), args.ner.max_grad_norm)
                        optimizer.step()

                        loss = loss.item()
                        running_loss.append(loss)
                        timestamps.append(time.time())
                        metrics = {
                            'Train Loss': loss,
                        }

                        if grad_steps % args.ner.print_interval == 0:
                            print(f'Step {grad_steps}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
                                f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')

                        if grad_steps % args.ner.eval_interval == 0:
                            eval_metrics = self.evaluate(val_dataset, args, n_samples=args.ner.eval_samples)
                            metrics.update(eval_metrics)
                            print(f'Step {grad_steps}:\t{eval_metrics}')

                        if args.wandb:
                            wandb.log(metrics, step=grad_steps)
        if args.wandb and args.ner.save_model and not args.debug:
            self.save()


    def evaluate(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        with Mode(self, 'eval'):
            dataloader = DataLoader(dataset,
                                    batch_size=args.ner.batch_size,
                                    num_workers=4,
                                    sampler=RandomSampler(dataset))
            losses = []
            preds = []
            labels = []
            p_pos = []
            for step, (input_ids, attention_mask, label) in enumerate(dataloader, start=1):
                label = label.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    output = self(input_ids, attention_mask).squeeze()
                    label = label.squeeze()
                    msk = (label != -1)
                    output = output[msk]
                    label = label[msk]
                    loss = F.cross_entropy(output, label)
                losses.append(loss.item())
                probs = F.softmax(output, dim=-1).cpu().numpy()
                pred = np.argmax(probs, axis=-1).flatten()
                label = label.cpu().numpy().flatten()
                preds.extend(pred)
                labels.extend(label)
                sample_p_pos = list(probs[:,1])
                p_pos.extend(sample_p_pos)
                if step * args.ner.batch_size >= n_samples:
                    break
            avg_loss = sum(losses) / len(losses)
            correct = np.equal(preds, labels)
            avg_acc = sum(correct) / len(preds)
            if args.ner.segmentation_only:
                avg_p_pos = sum(p_pos) / len(p_pos)
                p_pos_var = np.var(p_pos)
                pos = np.equal(preds, 1)
                true_pos = np.logical_and(pos, correct)
                false_neg = np.logical_and(1-pos,1-correct)
                f_score = sum(true_pos) / (sum(true_pos)
                                        + .5*(sum(pos) - sum(true_pos) + sum(false_neg)))
        metrics = {'Eval Loss': avg_loss,
                   'Eval Accuracy': avg_acc}
        if args.ner.segmentation_only:
            metrics.update({
                   'F-Score': f_score,
                   'Avg Positive Prob': avg_p_pos,
                   'Positive Prob Variance': p_pos_var})
            class_names = ['Divide', 'Continue']
        else:
            class_names = ner_token_names
        if args.wandb:
            confusion_matrix = wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=class_names)
            metrics.update({'Confusion Matrix': confusion_matrix})
        return metrics




