from functools import partial
from collections import namedtuple

import gym
import torch

from core.constants import argument_names
from core.dataset import EssayDataset
from core.models.essay_feedback import EssayModel
from utils.text import to_sentences
from utils.grading import get_labels, prediction_string, to_predictions

State = namedtuple('State', 'encoded_text predictions')

class SegmentationEnv(gym.Env):
    def __init__(self, essay_dataset, word_tokenizer, argument_classifier) -> None:
        super().__init__()

        self.dataset = essay_dataset
        self.word_tokenizer = word_tokenizer
        self.argument_classifier = argument_classifier
        self.essay = None
        self.encoded_essay_text = None
        self.word_idx = None
        self.done = None

    @property
    def state(self):
        return State(self.encoded_essay_text, self.predictions)

    def current_state_value(self):
        # logits = self.argument_classifier(self.essay.text, self.predictionstrings)
        # predictions = to_predictions(self.predictionstrings, logits, self.essay.essay_id)
        labels = self.essay.get_labels(self.predictions)
        predictions = [
            {'id': self.essay.essay_id,
             'class': argument_names[label],
             'predictionstring': prediction.pstring} for (prediction, label)
            in zip(self.predictions, labels)
        ]
        return self.essay.grade(predictions)['f_score']

    def reset(self):
        self.essay = self.dataset.random_essay()[0]
        self.predictions = []
        self.word_idx = 0
        self.done = False
        self.encoded_essay_text = self.word_tokenizer.encode(self.essay.text)
        return self.state

    def step(self, prediction):
        init_value = self.current_state_value()
        self.predictions.append(prediction)
        self.word_idx = prediction.stop
        if self.word_idx + 1 >= len(self.essay.words):
            self.done = True
        reward = self.current_state_value() - init_value
        return self.state, reward, self.done


class AssigmentEnv(gym.Env):
    def __init__(self, n_essays=None) -> None:
        super().__init__()
        self.actions = {
            0: self._move_up,
            1: self._move_down,
            2: self._merge,
            3: self._split,
            4: self._advance,
            5: partial(self._assign, 'Lead'),
            6: partial(self._assign, 'Position'),
            7: partial(self._assign, 'Claim'),
            8: partial(self._assign, 'Counterclaim'),
            9: partial(self._assign, 'Rebuttal'),
            10: partial(self._assign, 'Evidence'),
            11: partial(self._assign, 'Concluding Statement'),
            12: partial(self._assign, None),
            13: self._end
        }
        self.sentence_state = None
        self.argument_state = None
        self._position = None
        self.reward = None
        self.done = None
        print('Loading Dataset')
        self.dataset = EssayDataset(n_essays=n_essays)
        print('Dataset Loaded')
        self.max_sentences = 60
        self.max_args = 20

    def reset(self):
        self.done = False
        self.reward = 0
        self.essay = self.dataset.random_essay()[0]
        self.sentences = to_sentences(self.essay.text)
        self._position = 0
        self.sentence_state = None
        self.argument_state = None
        return self.state

    @property
    def position(self):
        position = torch.zeros(self.max_sentences)
        position[self._position] = 1
        return position

    @property
    def state(self):
        return self.position, self.sentence_state, self.argument_state

    def step(self, action):
        if self.done:
            raise RuntimeError('Environment is done, must be reset')
        if action not in self.actions:
            raise ValueError('Action not in available actions')
        else:
            func = self.actions[action]
            func()
        return self.state, self.reward, self.done

    def _move_up(self):
        if self._position < (self.max_sentences - 1):
            self._position += 1

    def _move_down(self):
        if self._position > 0:
            self._position -= 1

    def _merge(self):
        pass

    def _advance(self):
        pass

    def _split(self):
        pass

    def _assign(self, label):
        pass

    def _end(self):
        # self.reward = essay.grade(self.submission(), self.essay)
        self.done = True

    def submission(self):
        pass
