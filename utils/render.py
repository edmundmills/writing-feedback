import matplotlib.pyplot as plt

def plot_ner_output(tensor):
    ner_probs = tensor.squeeze().cpu().numpy().T
    plt.imshow(ner_probs, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()