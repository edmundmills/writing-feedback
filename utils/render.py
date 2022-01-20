import matplotlib.pyplot as plt
import torch

def plot_ner_output(tensor):
    if isinstance(tensor, torch.Tensor):
        ner_probs = tensor.squeeze().cpu().numpy()
    else:
        ner_probs = tensor
    ner_probs = ner_probs.T
    plt.imshow(ner_probs, cmap='hot', interpolation='nearest', aspect='auto',
               vmin=0, vmax=1)
    plt.show()