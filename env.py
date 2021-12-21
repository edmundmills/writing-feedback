from functools import partial

import gym
import torch

from dataset import ArgumentDataset
from utils import grade

class AssigmentEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.actions = {
            0: ('Up', self._move_up),
            1: ('Down', self._move_down),
            2: ('Merge', self._merge),
            3: ('Split', self._split),
            4: ('Lead', partial(self._assign, 'Lead')),
            5: ('Position', partial(self._assign, 'Position')),
            6: ('Claim', partial(self._assign, 'Claim')),
            7: ('Counterclaim', partial(self._assign, 'Counterclaim')),
            8: ('Rebuttal', partial(self._assign, 'Rebuttal')),
            9: ('Evidence', partial(self._assign, 'Evidence')),
            10: ('Concluding Statement', partial(self._assign, 'Concluding Statement')),
            11: ('Unlabel', partial(self._assign, None)),
            12: ('End', self._end)
        }
        self.sentence_state = None
        self.argument_state = None
        self._position = None
        self.essay = None
        self.reward = None
        self.done = None
        print('Loading Dataset')
        self.dataset = ArgumentDataset()
        print('Dataset Loaded')
        self.max_sentences = 30

    def reset(self):
        self.done = False
        self.reward = 0
        self.essay = self.dataset.random_essay()[0]
        self._position = 0
        self.sentence_state = None
        self.argument_state = None

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
            func = self.actions[action][1]
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

    def _assign(self, label):
        pass

    def _split(self):
        pass

    def _end(self):
        # self.reward = grade(self.submission(), self.essay)
        self.done = True

    def submission(self):
        pass
