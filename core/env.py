from functools import partial

import gym
import torch

from core.dataset import EssayDataset
from core.models.essay_feedback import EssayModel
from utils.text import to_sentences
from utils.grading import prediction_string, to_predictions

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
        return self.encoded_essay_text, self.predictionstrings

    @property
    def reward(self):
        reward = 0 if not self.done else self.current_state_value()
        return reward

    def current_state_value(self):
        logits = self.argument_classifier(self.essay.text, self.predictionstrings)
        predictions = to_predictions(self.predictionstrings, logits, self.essay.essay_id)
        return self.essay.grade(predictions)

    def reset(self):
        self.essay = self.dataset.random_essay()[0]
        self.predictionstrings = []
        self.word_idx = 0
        self.done = False
        self.encoded_essay_text = self.word_tokenizer.encode_plus(self.essay.text)
        return self.state

    def step(self, n_words):
        predictionstring = prediction_string(self.word_idx, self.word_idx + n_words - 1)
        self.predictionstrings.append(predictionstring)
        self.word_idx += n_words
        if self.word_idx + 1 >= len(self.essay.words):
            self.done = True
        return self.state, self.reward, self.done

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
