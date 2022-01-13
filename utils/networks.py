import math
from pathlib import Path

import torch
import torch.nn as nn
import wandb

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

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
    def __init__(self, d_model, max_seq_len=768):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(d_model, max_seq_len)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[i + 1, pos] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        self.pe = pe

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        n_d_elems = x.size(0)
        with torch.no_grad():
            x = x + self.pe[:n_d_elems,...]
        return x