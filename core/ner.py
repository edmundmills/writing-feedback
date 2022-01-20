from collections import deque
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import LongformerTokenizerFast, LongformerModel
import wandb

from utils.networks import MLP, Model, Mode
from core.constants import argument_names
from utils.render import plot_ner_output

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
        self.seg_only = args.segmentation_only
        self.n_outputs = 2 if args.segmentation_only else 10
        self.classifier = MLP(768,
                              self.n_outputs,
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
        n_essays, n_tokens, n_categories = probs.size()
        range_tensor = torch.arange(n_tokens, device=probs.device)
        range_tensor = range_tensor.repeat(n_essays, 1)
        word_idxs = range_tensor + range_tensor - word_ids - 1
        total_words = torch.max(word_ids, dim=-1, keepdim=True).values
        msk = torch.le(range_tensor, total_words + 1)
        msk = msk.unsqueeze(-1).repeat(1,1,n_categories)
        word_idxs = word_idxs.unsqueeze(-1).repeat(1,1,n_categories)
        probs = torch.gather(probs, dim=-2, index=word_idxs*msk)*msk
        return probs

    def forward(self, input_ids, attention_mask):
        encoded = self.transformer(input_ids, attention_mask=attention_mask)
        output = self.classifier(encoded.last_hidden_state)
        return output

    def inference(self, input_ids, attention_mask=None, word_ids=None):
        if self.feature_extractor:
            input_ids, attention_mask, word_ids = self.split_essay_tokens(input_ids)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        word_ids = word_ids.to(self.device)
        with torch.no_grad():
            self.eval()
            output = self(input_ids, attention_mask)
            output = self.collate_word_idxs(output, word_ids)
        if not self.seg_only:
            seg_output, class_output = torch.split(output, [2,8], dim=-1)
        else:
            seg_output = output
        seg_output = F.softmax(seg_output, dim=-1)
        _, seg_output = torch.chunk(seg_output, 2, dim=-1)
        if not self.seg_only:
            class_output = F.softmax(class_output, dim=-1)
            output = torch.cat((seg_output, class_output), dim=-1)
        else:
            output = seg_output
        attention_mask = attention_mask.unsqueeze(-1).repeat(1,1,output.size(-1))
        attention_mask = self.collate_word_idxs(attention_mask, word_ids)
        output = output * attention_mask - (1 - attention_mask)
        return output.cpu()

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
                seg_loss = 0
                class_loss = 0
                for idx, (input_ids, attention_mask, labels, _word_ids) in enumerate(dataloader):
                    step += 1
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device).squeeze()

                    output = self(input_ids, attention_mask).squeeze()
                    if not self.seg_only:
                        seg_output, class_output = torch.split(output, [2,8], dim=-1)
                        start_labels = labels[0]
                        class_labels = labels[1]
                    else:
                        seg_output = output
                        start_labels = labels
                    msk = (start_labels != -1)
                    seg_output = seg_output[msk]
                    start_labels = start_labels[msk]
                    seg_loss += F.cross_entropy(seg_output, start_labels)
                    if not self.seg_only:
                        class_output = class_output[msk]
                        class_labels = class_labels[msk]
                        class_loss += F.cross_entropy(class_output, class_labels)

                    if step % args.ner.grad_accumulation == 0 or idx == len(train_dataset) - 1:
                        grad_steps = step // args.ner.grad_accumulation
                        loss = seg_loss
                        if not self.seg_only:
                            loss += class_loss
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), args.ner.max_grad_norm)
                        optimizer.step()

                        loss = loss.item() / args.ner.grad_accumulation
                        running_loss.append(loss)
                        timestamps.append(time.time())
                        metrics = {
                            'Train Loss': loss,
                            'Segmentation/Train Loss': seg_loss.item()
                        }
                        if not self.seg_only:
                            metrics.update({
                                'Classification/Train Loss': class_loss.item()
                            })
                        loss = 0
                        seg_loss = 0
                        class_loss = 0

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
            seg_losses = []
            class_losses = []
            running_start_preds = []
            running_class_preds = []
            running_start_labels = []
            running_class_labels = []
            running_p_pos = []
            for step, (input_ids, attention_mask, labels, _word_ids) in enumerate(dataloader, start=1):
                labels = labels.to(self.device).squeeze()
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    output = self(input_ids, attention_mask).squeeze()
                    if not self.seg_only:
                        seg_output, class_output = torch.split(output, [2,8], dim=-1)
                        start_labels = labels[0]
                        class_labels = labels[1]
                    else:
                        seg_output = output
                        start_labels = labels
                    msk = (start_labels != -1)
                    seg_output = seg_output[msk]
                    start_labels = start_labels[msk]
                    seg_loss = F.cross_entropy(seg_output, start_labels)
                    loss = seg_loss
                    seg_losses.append(seg_loss.item())
                    if not self.seg_only:
                        class_output = class_output[msk]
                        class_labels = class_labels[msk]
                        class_loss = F.cross_entropy(class_output, class_labels)
                        loss += class_loss
                        class_losses.append(class_loss.item())
                losses.append(loss.item())
                start_probs = F.softmax(seg_output, dim=-1).cpu().numpy()
                start_preds = np.argmax(start_probs, axis=-1).flatten()
                start_labels = start_labels.cpu().numpy().flatten()
                running_start_preds.extend(start_preds)
                running_start_labels.extend(start_labels)
                p_pos = list(start_probs[:,1])
                running_p_pos.extend(p_pos)
                if not self.seg_only:
                    class_probs = F.softmax(class_output, dim=-1).cpu().numpy()
                    class_preds = np.argmax(class_probs, axis=-1).flatten()
                    class_labels = class_labels.cpu().numpy().flatten()
                    running_class_labels.extend(class_labels)
                    running_class_preds.extend(class_preds)
                if step * args.ner.batch_size >= n_samples:
                    break
        
        avg_loss = sum(losses) / len(losses)
        avg_seg_loss = sum(seg_losses) / len(seg_losses)
        seg_correct = np.equal(running_start_preds, running_start_labels)
        avg_seg_acc = sum(seg_correct) / len(running_start_preds)            
        avg_p_pos = sum(running_p_pos) / len(running_p_pos)
        p_pos_var = np.var(running_p_pos)
        pos = np.equal(running_start_preds, 1)
        true_pos = np.logical_and(pos, seg_correct)
        false_neg = np.logical_and(1-pos,1-seg_correct)
        seg_f_score = sum(true_pos) / (sum(true_pos)
                                + .5*(sum(pos) - sum(true_pos) + sum(false_neg)))
        metrics = {'Eval Loss': avg_loss,
                   'Segmentation/Eval Loss': avg_seg_loss,
                   'Segmentation/Eval Accuracy': avg_seg_acc,
                   'Segmentation/F-Score': seg_f_score,
                   'Segmentation/Avg Positive Prob': avg_p_pos,
                   'Segmentation/Positive Prob Variance': p_pos_var}
        
        if not self.seg_only:
            class_correct = np.equal(running_class_preds, running_class_labels)
            avg_class_acc = sum(class_correct) / len(running_class_preds)
            avg_class_loss = sum(class_losses) / len(class_losses)
            metrics.update({
                'Classification/Eval Accuracy': avg_class_acc,
                'Classification/Eval Loss': avg_class_loss,
            })
        
        if args.wandb:
            seg_confusion_matrix = wandb.plot.confusion_matrix(
                y_true=running_start_labels,
                preds=running_start_preds,
                class_names=['Divide', 'Continue'])
            metrics.update({'Segmentation/Confusion Matrix': seg_confusion_matrix})
            if not self.seg_only:
                class_confusion_matrix = wandb.plot.confusion_matrix(
                    y_true=running_class_labels,
                    preds=running_class_preds,
                    class_names=argument_names
                )
                metrics.update({'Classification/Confusion Matrix': class_confusion_matrix})
        return metrics




