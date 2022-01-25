from collections import deque
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import LongformerTokenizerFast, LongformerModel
import tqdm
import wandb

from utils.networks import MLP, Model, Mode
from utils.constants import de_num_to_type, ner_num_to_token
from utils.render import plot_ner_output


def get_pred_labels(predictions, num_words):
    tokens = []
    for pred in predictions:
        start_token = pred.label
        cont_token = start_token + 7 if start_token != 0 else 0
        tokens.append(start_token)
        tokens.extend([cont_token] * (len(pred.word_idxs) - 1))
    while len(tokens) < num_words:
        tokens.append(-1)
    tokens = np.array(tokens[:num_words], dtype=np.int8)
    return tokens


class NERTokenizer:
    def __init__(self, args):
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

    def make_ner_dataset(self, essay_dataset) -> TensorDataset:
        print('Making NER Dataset')
        input_ids = []
        attention_masks = []
        labels = []
        word_ids = []
        for essay in tqdm.tqdm(essay_dataset):
            encoded = self.encode(essay.text)
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            word_ids.append(encoded['word_id_tensor'])
            label_tokens = get_pred_labels(essay.correct_predictions, self.max_tokens)
            label_tokens = np.array([label_tokens[word_idx] if word_idx != None else -1
                                     for word_idx in encoded['word_ids']])
            labels.append(torch.LongTensor(label_tokens).squeeze())
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)
        word_ids = torch.stack(word_ids, dim=0)
        dataset = TensorDataset(input_ids, attention_masks, labels, word_ids)
        print('NER Dataset Created')
        return dataset


class NERModel(Model):
    def __init__(self, args) -> None:
        super().__init__()
        print('Loading NERModel')
        self.transformer = LongformerModel.from_pretrained(
            'allenai/longformer-base-4096').to(self.device)
        self.n_outputs = len(ner_num_to_token)
        self.classifier = MLP(768,
                              self.n_outputs,
                              args.linear_layers,
                              args.linear_layer_size,
                              dropout=0.1).to(self.device)
        print('NERModel Loaded')

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
        probs = torch.cat((probs[:,1:,:],
                           torch.zeros(n_essays, 1, n_categories, device=self.device)), dim=1)
        return probs

    def forward(self, input_ids, attention_mask):
        encoded = self.transformer(input_ids, attention_mask=attention_mask)
        output = self.classifier(encoded.last_hidden_state)
        return output

    def inference(self, input_ids, attention_mask, word_ids):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        word_ids = word_ids.to(self.device)
        with torch.no_grad():
            self.eval()
            output = self(input_ids, attention_mask)
            output = self.collate_word_idxs(output, word_ids)
        output = F.softmax(output, dim=-1)
        attention_mask = attention_mask.unsqueeze(-1).repeat(1,1,output.size(-1))
        attention_mask = self.collate_word_idxs(attention_mask, word_ids)
        output = output * attention_mask - (1 - attention_mask)
        return output.cpu()

    def infer_for_dataset(self, essay_dataset, tokenizer) -> None:
        print('Getting NER Probs')
        ner_probs = {}
        for essay in tqdm.tqdm(essay_dataset):
            encoded = tokenizer.encode(essay.text)
            probs = self.inference(encoded['input_ids'],
                                   encoded['attention_mask'],
                                   encoded['word_id_tensor'])
            ner_probs[essay.essay_id] = probs
        essay_dataset.ner_probs = ner_probs
        print('NER Probs Added to Dataset')
        return ner_probs

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
                loss = 0
                for idx, (input_ids, attention_mask, labels, _word_ids) in enumerate(dataloader):
                    step += 1
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device).squeeze()

                    output = self(input_ids, attention_mask).squeeze()
                    msk = (labels != -1)
                    output = output[msk]
                    labels = labels[msk]
                    loss += F.cross_entropy(output, labels)

                    if step % args.ner.grad_accumulation == 0 or idx == len(train_dataset) - 1:
                        grad_steps = step // args.ner.grad_accumulation
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), args.ner.max_grad_norm)
                        optimizer.step()

                        loss = loss.item() / args.ner.grad_accumulation
                        running_loss.append(loss)
                        timestamps.append(time.time())
                        metrics = {
                            'Train Loss': loss,
                        }
                        loss = 0

                        if grad_steps % args.ner.print_interval == 0:
                            print(f'Step {grad_steps}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
                                f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')

                        if grad_steps % args.ner.eval_interval == 0:
                            eval_metrics = self.evaluate(val_dataset, args, n_samples=args.ner.eval_samples)
                            metrics.update(eval_metrics)
                            print(f'Step {grad_steps}:\t{eval_metrics}')

                        if args.wandb:
                            wandb.log(metrics, step=grad_steps)
        print('Training Complete')


    def evaluate(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        with Mode(self, 'eval'):
            dataloader = DataLoader(dataset,
                                    batch_size=args.ner.batch_size,
                                    num_workers=4,
                                    sampler=RandomSampler(dataset))
            losses = []
            running_preds = []
            running_labels = []
            for step, (input_ids, attention_mask, labels, _word_ids) in enumerate(dataloader, start=1):
                labels = labels.to(self.device).squeeze()
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    output = self(input_ids, attention_mask).squeeze()
                    msk = (labels != -1)
                    output = output[msk]
                    labels = labels[msk]
                    loss = F.cross_entropy(output, labels)
                    losses.append(loss.item())
                probs = F.softmax(output, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1).flatten()
                running_preds.extend(preds)
                running_labels.extend(labels.cpu().numpy().flatten())
                if step * args.ner.batch_size >= n_samples:
                    break
        
        avg_loss = sum(losses) / len(losses)
        correct = np.equal(running_preds, running_labels)
        avg_acc = sum(correct) / len(running_preds)            
        metrics = {'Eval Loss': avg_loss,
                   'Eval Accuracy': avg_acc}
                
        if args.wandb:
            seg_confusion_matrix = wandb.plot.confusion_matrix(
                y_true=running_labels,
                preds=running_preds,
                class_names=ner_num_to_token)
            metrics.update({'Confusion Matrix': seg_confusion_matrix})
        return metrics




