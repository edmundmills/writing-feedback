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