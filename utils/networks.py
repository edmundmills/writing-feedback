from collections import defaultdict, deque
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import wandb


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    def save(self, model_name):
        model_dir = Path('models') / self.__class__.__name__
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f'{model_name}.pth' 
        print(f'Saving model as {model_file}')
        torch.save(self.state_dict(), model_file)
        print(f'Model Saved.')

    def load(self, model_name):
        model_file = Path('models') / self.__class__.__name__ / f'{model_name}.pth'
        print(f'Loading model from {model_file}')
        self.load_state_dict(torch.load(model_file))
        print('Model Loaded')

    def make_dataloader(self, dataset, args):
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           num_workers=4,
                                           shuffle=True)

    def make_optimizer(self, args):
        return torch.optim.AdamW(self.parameters(), args.learning_rate)

    def loss(self, sample, eval=False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def update_params(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def process_eval_metrics(self, metrics:Dict[str, List]):
        raise NotImplementedError

    def learn(self, train_dataset, val_dataset, args):
        dataloader = self.make_dataloader(train_dataset, args)
        optimizer = self.make_optimizer(args)
        step = 0
        running_loss = deque(maxlen=args.print_interval)
        timestamps = deque(maxlen=args.print_interval)

        with Mode(self, 'train'):
            for epoch in range(1, args.epochs + 1):
                print(f'Starting epoch {epoch}')
                for sample in dataloader:
                    step += 1
                    loss, metrics = self.loss(sample)
                    self.update_params(optimizer, loss)
                    loss = loss.item()
                    running_loss.append(loss)
                    timestamps.append(time.time())

                    if step % args.print_interval == 0:
                        print(f'Step {step}:\t Loss: {sum(running_loss)/len(running_loss):.3f}'
                            f'\t Rate: {len(timestamps)/(timestamps[-1]-timestamps[0]):.2f} It/s')

                    if step % args.eval_interval == 0:
                        eval_metrics = self.evaluate(val_dataset, args, n_samples=args.eval_samples)
                        metrics.update(eval_metrics)
                        print(f'Step {step}:\t{eval_metrics}')

                    wandb.log(metrics, step=step)
        print('Training Complete')

    def evaluate(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        dataloader = self.make_dataloader(dataset, args)
        metrics = defaultdict(list)
        with Mode(self, 'eval'):
            for idx, sample in enumerate(dataloader, start=1):
                with torch.no_grad():
                    loss, sample_metrics = self.loss(sample, eval=True)
                for k, v in sample_metrics.items():
                    metrics[k].append(v)
                if idx * args.batch_size >= n_samples:
                    break
        metrics = self.process_eval_metrics(metrics)
        return metrics


class Mode:
    def __init__(self, module, mode:str) -> None:
        self.module = module
        self.mode = mode
        self.initial_mode_training = self.module.training
    
    def __enter__(self):
        if self.mode == 'train':
            self.module.train()
        elif self.mode == 'eval':
            self.module.eval()            

    def __exit__(self, *args, **kwargs):
        if self.initial_mode_training:
            self.module.train()
        else:
            self.module.eval()

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, layer_size,
                 output_mod=None, dropout:float=None):
        super().__init__()
        if n_layers == 1:
                layers = [nn.Linear(n_inputs, n_outputs)]
        else:
            layers = [
                nn.Linear(n_inputs, layer_size),
                nn.ReLU(),
            ]
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(layer_size, layer_size))
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(layer_size, n_outputs))
        if output_mod is not None:
            layers.append(output_mod)
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, features=768, device='cpu'):
        super().__init__()
        self.features = features
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(seq_len, features)
        for pos in range(features):
            for i in range(0, seq_len, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i)/seq_len)))
                pe[i + 1, pos] = math.cos(pos / (10000 ** ((2 * (i + 1))/seq_len)))
        self.pe = pe.to(device)
        self.pe.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            # make embeddings relatively larger
            x = x * math.sqrt(self.features)
            #add constant to embedding
            x = x + self.pe
        return x