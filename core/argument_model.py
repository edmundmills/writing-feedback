import numpy as np
from sentence_transformers import SentenceTransformer
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

    def eval(self, dataset, args, n_samples=None):
        n_samples = n_samples or len(dataset)
        self.encoder.eval()
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                drop_last=True,
                                sampler=SequentialSampler(dataset))
        losses = []
        accuracy = []
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
            accuracy.append(sum(np.equal(pred, label)) / len(sample))
            if (step + 1) * args.batch_size >= n_samples:
                break
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accuracy) / len(accuracy)
        self.encoder.train()
        return {'Eval Loss': avg_loss, 'Eval Accuracy': avg_acc}
    
    def train(self, train_dataset, val_dataset, args):
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
            print(f'Starting Epoch {epoch}')
            for step, (samples, labels) in enumerate(dataloader):
                labels = labels.to(self.device)

                encoded_samples = self.encode(samples)
                logits = self.type_classifier(encoded_samples)

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                d_step = step + 1
                metrics = {'Train Loss': loss}

                if d_step % 50 == 0:
                    print(f'Step {d_step}:\t Loss: {loss:.3f}')

                if d_step % 50 == 0:
                    eval_metrics = self.eval(val_dataset, args, n_samples=(args.batch_size * 4))
                    metrics.update(eval_metrics)
                    print(f'Step {d_step}:\t{eval_metrics}')

                wandb.log(metrics, step=step)


