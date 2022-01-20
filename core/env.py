from functools import partial

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from core.constants import argument_names
from core.dataset import EssayDataset
from core.essay import Prediction
from utils.text import to_sentences
from utils.grading import to_tokens


env_classes = {}

def register_env(cls):
    env_classes.update({cls.__name__: cls})
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)
    return wrapper


class SegmentationEnv(gym.Env):
    def __init__(self, essay_dataset, env_args) -> None:
        super().__init__()
        self.dataset = essay_dataset
        self.max_d_elems = env_args.num_d_elems
        self.max_words = env_args.num_words
        self.essay = None
        self.ner_probs_space = spaces.Box(low=0, high=1,
                                          shape=(self.max_words, 9))
        self.segmentation_tokens_space = spaces.Box(low=-1, high=1,
                                                    shape=(self.max_words,))
        self.classification_token_space = spaces.Box(low=0, high=1,
                                                     shape=(self.max_d_elems, 8))
        self.observation_space = spaces.Dict({
            'ner_probs': self.ner_probs_space,
            'seg_tokens': self.segmentation_tokens_space,
            'class_tokens': self.classification_token_space
        })

    @staticmethod
    def make(n_envs, essay_dataset, env_args):
        cls = env_classes[env_args.class_name]
        if n_envs == 1:
            print(f'Making Non-Vectorized {cls.__name__} Environment')
            env = cls(essay_dataset, env_args)
            env = Monitor(env)
            print('Env Created')
            return env
        print(f'Making Vectorized {cls.__name__} Environment')
        dataset_fracs = [1 / n_envs] * n_envs
        datasets = essay_dataset.split(dataset_fracs)
        def make_env(dataset):
            def _init():
                env = cls(dataset, env_args)
                return env
            return _init
        venv = SubprocVecEnv([make_env(ds) for ds in datasets])
        venv = VecMonitor(venv, filename=f'./log/test')
        print('Vectorized env created')
        return venv

    @property
    def seg_tokens(self):
        return to_tokens(self.predictions,
                         num_words=self.max_words)[0]

    @property
    def class_tokens(self):
        labels = [pred.label for pred in self.predictions[:self.max_d_elems]]
        class_tokens = np.zeros((self.max_d_elems, 8), dtype=np.int8)
        class_tokens[np.arange(len(labels)), labels] = 1
        return class_tokens

    @property
    def state(self):
        state = {
            'seg_tokens': self.seg_tokens,
            'class_tokens': self.class_tokens,
            'ner_probs': self.ner_probs
        }
        return state

    def current_state_value(self):
        labels = self.essay.get_labels(self.predictions)
        predictions = [
            {'id': self.essay.essay_id,
             'class': argument_names[label],
             'predictionstring': prediction.pstring} for (prediction, label)
            in zip(self.predictions, labels)
        ]
        return self.essay.grade(predictions)['f_score']


    def reset(self, essay_id=None):
        if essay_id is None:
            self.essay = self.dataset.random_essay()[0]
        else:
            self.essay = self.dataset.get_by_id(essay_id)
        self.predictions = []
        self.done = False
        self.ner_probs = self.dataset.ner_probs[self.essay.essay_id].squeeze(0).numpy()
        self.steps = 0
        self.env_init_value = self.current_state_value()
        return self.state


@register_env
class SequencewiseEnv(SegmentationEnv):
    def __init__(self, essay_dataset, args) -> None:
        super().__init__(essay_dataset, args)
        self.continuous = args.continuous
        self.action_space_dim = args.action_space_dim
        if args.continuous:
            self.action_space = spaces.Box(-1, 1, (1,))
        else:
            self.action_space = spaces.Discrete(self.action_space_dim)

    def reset(self, *args, **kwargs):
        self.word_idx = 0
        return super().reset(*args, **kwargs)

    def step(self, action:int):
        if self.continuous:
            word_step = int((action + 1) * self.action_space_dim/2 + 1)
        else:
            word_step = int(action)
        init_value = self.current_state_value()
        pred_end = min(self.word_idx + word_step - 1, len(self.essay.words) - 1)
        self.predictions.append(Prediction(self.word_idx, pred_end, 0, self.essay.essay_id))
        self.word_idx += word_step
        self.steps += 1
        if self.word_idx + 1 >= min(len(self.essay.words), self.max_words) or self.steps >= self.max_d_elems:
            self.done = True
            print(self.steps)
        reward = self.current_state_value() - init_value
        info = {}
        return self.state, reward, self.done, info


@register_env
class SplitterEnv(SegmentationEnv):
    def __init__(self, essay_dataset, env_args) -> None:
        super().__init__(essay_dataset, env_args)
        self._seg_tokens = np.zeros(self.max_words, dtype=np.int8)
        self.action_space = spaces.Discrete(self.max_words)

    @property
    def seg_tokens(self):
        return self._seg_tokens

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.essay_num_words = min(self.max_words, len(self.essay.words))
        self._seg_tokens = np.concatenate((
            np.zeros(self.essay_num_words, dtype=np.int8),
            -np.ones(self.max_words - self.essay_num_words, dtype=np.int8)))
        self._seg_tokens[0] == 1
        self.predictions = self.pred_tokens_to_preds()
        self.env_init_value = self.current_state_value()
        self.steps = 0
        return self.state

    def pred_tokens_to_preds(self):
        start = 0
        preds = []
        for idx, x in enumerate(self.seg_tokens[1:], start=1):
            if x == -1:
                preds.append(Prediction(start, idx - 1, 0, self.essay.essay_id))
                break
            elif x == 1:
                preds.append(Prediction(start, idx, 0, self.essay.essay_id))
                start = idx + 1
        return preds              

    def step(self, action):
        init_value = self.current_state_value()
        self.steps += 1
        if int(action) < self.essay_num_words - 1:
            self._seg_tokens[int(action)] = 1
            self.predictions = self.pred_tokens_to_preds()
        self.done = (int(action) == self.max_words - 1) or self.steps >= self.max_d_elems - 1
        reward = self.current_state_value() - init_value
        info = {}
        return self.state, reward, self.done, info


@register_env
class DividerEnv(SegmentationEnv):
    def __init__(self, essay_dataset, env_args) -> None:
        super().__init__(essay_dataset, env_args)
        self.action_space = spaces.MultiDiscrete([3]*(self.max_d_elems - 1) + [2])

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.predictions = self._initial_predictions()
        self.env_init_value = self.current_state_value()
        self.steps = 0
        return self.state

    def step(self, action):
        init_value = self.current_state_value()
        distance = 4
        self.steps += 1
        for idx, act in enumerate(action[:-2]):
            new_stop = min(self.max_words - 1, self.predictions[idx].stop + (distance * (int(act) - 1)))
            self.predictions[idx].stop = new_stop
            new_start = max(0, self.predictions[idx+1].start + (distance * (int(act) - 1)))
            self.predictions[idx+1].start = new_start
        self.done = bool(action[-1]) and self.steps >= 10
        reward = self.current_state_value() - init_value
        info = {}
        return self.state, reward, self.done, info

    def _initial_predictions(self):
        preds = []
        start = 0
        for _ in range(self.max_d_elems):
            label = -1
            stop = min(start + self.max_words // self.max_d_elems, self.max_words - 1)
            preds.append(Prediction(start, stop, label, self.essay.essay_id))
            start = stop + 1
        preds[-1].stop = self.max_words - 1
        return preds


@register_env
class WordwiseEnv(SegmentationEnv):
    def __init__(self, essay_dataset, env_args) -> None:
        super().__init__(essay_dataset, env_args)
        self.action_space = spaces.MultiDiscrete(
            [2] + [2] * self.max_words + [len(argument_names)] * self.max_d_elems)
        self.observation_space = spaces.Dict({
            'ner_probs': self.ner_probs_space,
            'prev_segmentation': spaces.Box(low=0, high=1, shape=(self.max_words,)),
            'prev_d_elem_tokens': spaces.Box(
                low=0, high=100000,
                shape=(self.max_d_elems, len(argument_names)))
        })

    @property
    def state(self):
        state = {
            'ner_probs': self.ner_probs,
            'prev_segmentation': self.prev_segmentation,
            'prev_d_elem_tokens': self.prev_d_elem_tokens,
        }
        return state

    def step(self, action):
        init_value = self.current_state_value()
        self.done = bool(action[0])
        self.prev_segmentation = action[1:self.max_words+1]
        classifications = action[self.max_words+1:]
        # make predictions
        # save prev_d_elem_tokens

        reward = self.current_state_value() - init_value
        info = {}
        return self.state, reward, self.done, info

    def reset(self):
        self.prev_segmentation = np.zeros(self.max_words)
        self.prev_d_elem_tokens = np.zeros((self.max_d_elems, len(argument_names)))
        return super().reset()


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
