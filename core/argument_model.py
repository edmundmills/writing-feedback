from collections import deque
from pathlib import Path
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import wandb

from core.dataset import argument_names

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, layer_size, output_mod=None):
        super().__init__()
        if n_layers == 1:
                layers = [nn.Linear(n_inputs, n_outputs)]
        else:
            layers = [
                nn.Linear(n_inputs, layer_size),
                nn.ReLU(),
            ]
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(layer_size, layer_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(layer_size, n_outputs))
        if output_mod is not None:
            layers.append(output_mod)
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class ArgumentClassifier(nn.Module):
    def __init__(self):
        super().__init__()        
        self.layer_size = 512
        self.categories = 8 # includes none
        self.mlp = MLP(n_inputs=768,
                       n_outputs=self.categories,
                       n_layers=2,
                       layer_size=512,
                       output_mod=nn.Softmax(dim=1))
  
    def forward(self, encoded_argument):
        return self.mlp(encoded_argument)


class PolarityAssesor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(n_inputs=768*2,
                       n_outputs=1,
                       n_layers=2,
                       layer_size=512,
                       output_mod=nn.Tanh())

    def forward(self, encoded_argument1, encoded_argument2):
        input = torch.cat((encoded_argument1, encoded_argument2), dim=1)
        return self.mlp(input)


class MergeAssesor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(n_inputs=768*2,
                       n_outputs=2,
                       n_layers=2,
                       layer_size=512,
                       output_mod=nn.Softmax(dim=1))

    def forward(self, encoded_argument1, encoded_argument2):
        input = torch.cat((encoded_argument1, encoded_argument2), dim=1)
        return self.mlp(input)


class ArgumentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Loading Argument Model')
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
        self.type_classifier = ArgumentClassifier().to(self.device)
        self.polarity = PolarityAssesor().to(self.device)
        self.merge_assesment = MergeAssesor().to(self.device)
        self.split_assesment = None
        print('Argument Model Loaded.')

    def encode(self, argument):
        return self.encoder.encode(argument, convert_to_numpy=False, convert_to_tensor=True)

    def evaluate(self, class_dataset, polarity_dataset, args, n_samples=None):
        class_eval_metrics = self.eval_classifier(class_dataset, args, n_samples)
        polarity_eval_metrics = self.eval_polarity(polarity_dataset, args, n_samples)
        return {**class_eval_metrics, **polarity_eval_metrics}

    def eval_classifier(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        self.encoder.eval()
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(dataset))
        losses = []
        preds = []
        labels = []
        for step, (sample, label) in enumerate(dataloader):
            gpu_label = label.to(self.device)
            with torch.no_grad():
                encodings = self.encode(sample)
                logits = self.type_classifier(encodings)
                loss = F.cross_entropy(logits, gpu_label)
            losses.append(loss.item())
            logits = logits.cpu().numpy()
            pred = np.argmax(logits, axis=1).squeeze()
            label = label.numpy().squeeze()
            preds.extend(pred)
            labels.extend(label)
            if (step + 1) * args.batch_size >= n_samples:
                break
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(np.equal(pred, label)) / len(pred)
        self.encoder.train()
        confusion_matrix = wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=argument_names)
        return {'Type Classifier/Eval Loss': avg_loss,
                'Type Classifier/Eval Accuracy': avg_acc,
                'Type Classifier/Confusion Matrix': confusion_matrix}

    def eval_polarity(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        self.encoder.eval()
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(dataset))
        losses = []
        labels = []
        preds = []
        for step, (sample, label) in enumerate(dataloader):
            gpu_label = label.to(self.device)
            with torch.no_grad():
                encodings1 = self.encode(sample[0])
                encodings2 = self.encode(sample[1])
                output = self.polarity(encodings1, encodings2).squeeze()
                loss = F.mse_loss(output, gpu_label)
            losses.append(loss.item())
            output = output.cpu().numpy()
            def polarity_eval(x):
                if x > (1/3):
                    return 2
                elif x < (-1/3):
                    return 0
                else:
                    return 1
            preds.extend([polarity_eval(pred) for pred in output])
            labels.extend([polarity + 1 for polarity in label.numpy().squeeze()])
            if (step + 1) * args.batch_size >= n_samples:
                break
        avg_loss = sum(losses) / len(losses)
        accuracy = sum(np.equal(preds, labels)) / len(preds)
        self.encoder.train()
        confusion_matrix = wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=['None', 'Supporting', 'Contradicting'])
        return {'Polarity/Eval Loss': avg_loss,
                'Polarity/Eval Accuracy': accuracy,
                'Polarity/Confusion Matrix': confusion_matrix}
    
    def train(self,
              class_train_dataset, class_val_dataset,
              polarity_train_dataset, polarity_val_dataset,
              args):
        epochs = args.epochs
        batch_size = args.batch_size

        self.encoder.train()

        optimizer = AdamW([{'params': self.encoder.parameters(), 'lr': args.encoder_lr},
                          {'params': self.type_classifier.parameters(), 'lr': args.type_lr},
                          {'params': self.polarity.parameters(), 'lr': args.polarity_lr}],
                          lr=args.encoder_lr)
        class_dataloader = DataLoader(class_train_dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(class_train_dataset))
        polarity_dataloader = DataLoader(polarity_train_dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(polarity_train_dataset))
        
        polarity_iter = iter(polarity_dataloader)

        running_loss = deque(maxlen=args.print_interval)
        timestamps = deque(maxlen=args.print_interval)

        step = 0
        for epoch in range(1, epochs + 1):
            print(f'Starting Epoch {epoch}')
            for class_samples, class_labels in class_dataloader:
                step += 1
                try:
                    polarity_samples, polarity_labels = next(polarity_iter)
                except StopIteration:
                    polarity_iter = iter(polarity_dataloader)
                    polarity_samples, polarity_labels = next(polarity_iter)

                class_labels = class_labels.to(self.device)
                polarity_labels = polarity_labels.to(self.device)

                encoded_class_samples = self.encode(class_samples)
                class_logits = self.type_classifier(encoded_class_samples)
                class_loss = F.cross_entropy(class_logits, class_labels)

                encodings1 = self.encode(polarity_samples[0])
                encodings2 = self.encode(polarity_samples[1])
                output = self.polarity(encodings1, encodings2).squeeze()
                polarity_loss = F.mse_loss(output, polarity_labels)

                loss = class_loss + polarity_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                loss = loss.item()
                running_loss.append(loss)
                timestamps.append(time.time())
                metrics = {
                    'Type Classifier/Train Loss': class_loss.item(),
                    'Polarity/Train Loss': polarity_loss.item(),
                    'Total Train Loss': loss,
                }

                if step % args.print_interval == 0:
                    print(f'Step {step}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
                          f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')

                if step % args.eval_interval == 0:
                    eval_metrics = self.evaluate(class_val_dataset, polarity_val_dataset, args, n_samples=(args.batch_size * args.batches_per_eval))
                    metrics.update(eval_metrics)
                    print(f'Step {step}:\t{eval_metrics}')

                wandb.log(metrics, step=step)
        self.save()

    def save(self):
        model_dir = Path('models') / self.__class__.__name__
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f'{wandb.run.name}.pth' 
        print(f'Saving model as {model_file}')
        torch.save(self.state_dict(), model_file)
        print(f'Model Saved.')

    def load(self, model_name):
        model_file = Path('models') / self.__class__.__name__ / f'{model_name}.pth'
        print(f'Loading model from {model_file}')
        self.load_state_dict(torch.load(model_file))
        print('Model Loaded')


