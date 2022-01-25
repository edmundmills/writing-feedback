import itertools
from termcolor import colored
from typing import List

import matplotlib.pyplot as plt
from matplotlib import cm
import torch

from utils.constants import ner_num_to_token



ner_color_map = {
    0: ('grey',),
    1: ('grey', 'on_white'),
    2: ('white', 'on_red'),
    3: ('white', 'on_blue'),
    4: ('white', 'on_magenta'),
    5: ('white', 'on_cyan'),
    6: ('white', 'on_green'),
    7: ('white', 'on_yellow'),
    8: ('white',),
    9: ('red',),
    10: ('blue',),
    11: ('magenta',),
    12: ('cyan',),
    13: ('green',),
    14: ('yellow',),
    -1: ('yellow', 'on_red')
}

start_color_map = {
    False: 'grey',
    True: 'white'
}

def plot_ner_output(tensor, segment_lens:List[int]=None):
    if isinstance(tensor, torch.Tensor):
        ner_probs = tensor.squeeze().cpu().numpy()
    else:
        ner_probs = tensor
    ner_probs = ner_probs.T
    plt.imshow(ner_probs, cmap='hot', interpolation='nearest', aspect='auto',
               vmin=0, vmax=1)
    if segment_lens:
        seg_x = list(itertools.accumulate(segment_lens))
        plt.vlines(seg_x, 0, 13)
    plt.show()


def rgb(rgb_color):
    r, g, b, *_ = rgb_color
    return f"\u001b[38;2;{r};{g};{b}m"

end_color = '\033[0m'

def colormapstr(list_to_print):
    cmap = cm.get_cmap('plasma')
    colormap = ''
    for y in list_to_print:
        rgb_color = cmap(y, bytes=True)
        color = rgb(rgb_color)
        colormap += color + u"\u2588" + end_color
    return colormap


class EssayRenderer:
    def __init__(self) -> None:
        self.num_words = 1024

    def render(self, essay, segment_lens:List=None, predictions:List=None):
        print('#'*80)
        print(f'Essay ID: {essay.essay_id}')
        print(f'Num Words: {len(essay)}')
        correct_ner_tokens = essay.ner_labels(num_words=self.num_words)
        if segment_lens:
            seg_x = list(itertools.accumulate(segment_lens))
        if predictions is not None:
            guess_ner_tokens = essay.ner_labels(num_words=self.num_words, predictions=predictions)
        else:
            guess_ner_tokens = [0] * self.num_words
        for idx, word in enumerate(essay.words):
            correct_token = correct_ner_tokens[idx]
            guess_token = guess_ner_tokens[idx]
            correct_token_str = str(correct_token)
            if len(correct_token_str) == 1:
                correct_token_str += ' '
            guess_token_str = str(guess_token)
            if len(guess_token_str) == 1:
                guess_token_str += ' '

            seg = idx in seg_x
            render_word = f'{word}\t' + \
                  f'{colored(correct_token_str, *ner_color_map[correct_token])}' + \
                  '\t'.expandtabs(10) + \
                  f'{colored(int(seg), start_color_map[seg])}' + \
                  '\t'.expandtabs(10) + \
                  f'{colored(guess_token_str, *ner_color_map[guess_token])}' + \
                  '\t'.expandtabs(10)
            if essay.ner_probs is not None:
                render_word += colormapstr(essay.ner_probs[0,idx, 1:8].squeeze().tolist())
                render_word += ' '
                render_word += colormapstr(essay.ner_probs[0,idx,0:1].tolist())
                render_word += colormapstr(essay.ner_probs[0,idx, 8:].squeeze().tolist())
            print(render_word.expandtabs(20))
        if predictions:
            print(essay.grade(predictions))
        print('\n')