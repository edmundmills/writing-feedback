from functools import partial

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from core.constants import argument_names
from core.dataset import EssayDataset
from core.essay import Prediction
from utils.text import to_sentences
from utils.grading import to_tokens


class SegmentationEnv(gym.Env):
    def __init__(self, essay_dataset, word_tokenizer, d_elem_classifier, args) -> None:
        super().__init__()
        self.dataset = essay_dataset
        self.word_tokenizer = word_tokenizer
        self.d_elem_classifier = d_elem_classifier
        self.essay = None
        self.encoded_essay_text = None
        self.done = None
        self.max_words = word_tokenizer.max_tokens
    
    @classmethod
    def make_vec(cls, n_envs, essay_dataset, word_tokenizer, d_elem_classifier, env_args):
        print('Making Vectorized Environment')
        dataset_fracs = [1 / n_envs] * n_envs
        print(dataset_fracs)
        datasets = essay_dataset.split(dataset_fracs)
        def make_env(dataset):
            def _init():
                env = cls(dataset, word_tokenizer, d_elem_classifier, env_args)

                return env
            return _init
        venv = SubprocVecEnv([make_env(ds) for ds in datasets])
        venv = VecMonitor(venv, filename='./log/test')
        print('Vectorized env created')
        return venv

    @property
    def prediction_tokens(self):
        return to_tokens(self.predictions,
                         num_words=min(self.max_words, self.max_words))

    @property
    def state(self):
        raise NotImplementedError

    def current_state_value(self):
        labels = self.essay.get_labels(self.predictions)
        predictions = [
            {'id': self.essay.essay_id,
             'class': argument_names[label],
             'predictionstring': prediction.pstring} for (prediction, label)
            in zip(self.predictions, labels)
        ]
        return self.essay.grade(predictions)['f_score']


    def reset(self):
        if self.essay:
            print(self.current_state_value())
        self.essay = self.dataset.random_essay()[0]
        self.predictions = []
        self.done = False
        encoded = self.word_tokenizer.encode(self.essay.text)
        self.encoded_essay_text = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']
        self.word_ids = encoded.word_ids()
        self.essay_tokens = torch.cat((self.encoded_essay_text,
                                         self.attention_mask), dim=0).numpy()
        return self.state



class WordwiseEnv(SegmentationEnv):
    def __init__(self, essay_dataset, word_tokenizer, d_elem_classifier, env_args) -> None:
        super().__init__(essay_dataset, word_tokenizer, d_elem_classifier, env_args)
        self.action_space = spaces.Discrete(self.max_words)
        self.observation_space = spaces.Dict({
            'essay_tokens': spaces.Box(low=0, high=100000, shape=(2, self.max_words), dtype=np.int32),
            'pred_tokens': spaces.Box(low=-1, high=1, shape=(self.max_words,), dtype=np.int8),
            'classification_probs': spaces.Box(
                low=0, high=1,
                shape=(d_elem_classifier.max_d_elems, len(argument_names)))
        })

    @property
    def state(self):
        state = {
            'essay_tokens': self.essay_tokens,
            'pred_tokens': self.prediction_tokens,
            'classification_probs': self.classification_probs()
        }
        return state

    def classification_probs(self):
        pass

    def step(self, action):
        init_value = self.current_state_value()
        reward = self.current_state_value() - init_value
        info = {}
        return self.state, reward, self.done, info

    def reset(self):
        return super().reset()


class SequencewiseEnv(SegmentationEnv):
    def __init__(self, essay_dataset, word_tokenizer, d_elem_classifier, args) -> None:
        super().__init__(essay_dataset, word_tokenizer, d_elem_classifier, args)
        self.action_space = spaces.Discrete(args.action_space_dim)
        self.observation_space = spaces.Dict({
            'essay_tokens': spaces.Box(low=0, high=100000, shape=(2, self.max_words), dtype=np.int32),
            'pred_tokens': spaces.Box(low=-1, high=1, shape=(self.max_words,), dtype=np.int8)
        })

    def reset(self):
        self.word_idx = 0
        return super().reset()

    @property
    def state(self):
        state = {
            'essay_tokens': self.essay_tokens,
            'pred_tokens': self.prediction_tokens
        }
        return state

    def step(self, action:int):
        action = action + 2 # Segments must be at least 2 words long
        init_value = self.current_state_value()
        pred_end = min(self.word_idx + action - 1, len(self.essay.words) - 1)
        self.predictions.append(Prediction(self.word_idx, pred_end, -1, self.essay.essay_id))
        self.word_idx += action
        if self.word_idx + 1 >= min(len(self.essay.words), self.max_words):
            self.done = True
        reward = self.current_state_value() - init_value
        info = {}
        return self.state, reward, self.done, info


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
