from sentence_transformers import SentenceTransformer
from torch.nn.modules.activation import Softmax
from transformers import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb


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
        self.mlp = MLP(n_inputs=756,
                       n_outputs=self.categories,
                       n_layers=2,
                       layer_size=512,
                       output_mod=nn.Softmax())
  
    def forward(self, encoded_argument):
        return self.mlp(encoded_argument)


class PolarityAssesor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(n_inputs=756*2,
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
        self.mlp = MLP(n_inputs=756*2,
                       n_outputs=2,
                       n_layers=2,
                       layer_size=512,
                       output_mod=nn.Softmax())

    def forward(self, encoded_argument1, encoded_argument2):
        input = torch.cat((encoded_argument1, encoded_argument2), dim=1)
        return self.mlp(input)


class ArgumentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
        self.type_classifier = ArgumentClassifier().to(self.device)
        self.polarity = PolarityAssesor().to(self.device)
        self.merge_assesment = MergeAssesor().to(self.device)
        self.split_assesment = None

    def encode(self, argument):
        return self.encoder.encode(argument)

    def train(self, train_dataset, validation_dataset, args):
        epochs = args.epochs
        lr = args.lr
        batch_size = args.batch_size

        self.encoder.train()

        optimizer = AdamW(self.parameters(), lr=lr)
        dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=RandomSampler(train_dataset))
        for epoch in range(1, epochs + 1):
            print(f'Starting epoch {epoch}')
            for samples, labels in dataloader:
                labels = labels.to(self.device)

                encoded_samples = self.encode(samples)
                logits = self.classifier(encoded_samples)

                loss = F.cross_entropy_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                wandb.log({'loss': loss})

