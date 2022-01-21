from functools import partial

import gym
from gym import spaces
import numpy as np
from pandas import merge
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from core.constants import argument_names, argument_types
from core.dataset import EssayDataset
from core.essay import Prediction
from core.segmentation import segment_ner_probs
from utils.text import to_sentences
from utils.grading import to_tokens
from utils.render import plot_ner_output


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
        self.grade_classifications = env_args.grade_classifications
        self.ner_probs_space = spaces.Box(low=0, high=1,
                                          shape=(self.max_words, 9))
        self.segmentation_tokens_space = spaces.Box(low=-1, high=1,
                                                    shape=(self.max_words,))
        self.classification_token_space = spaces.Box(low=-1, high=1,
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
            env = Monitor(env, filename=f'./log/test', info_keywords=('Score',))
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
        venv = VecMonitor(venv, filename=f'./log/test', info_keywords=('Score',))
        print('Vectorized env created')
        return venv

    @property
    def seg_tokens(self):
        return to_tokens(self.predictions,
                         num_words=self.max_words)[0]

    @property
    def class_tokens(self):
        labels = [pred.label for pred in self.predictions[:self.max_d_elems]]
        class_tokens = np.zeros((len(self.predictions), 8), dtype=np.int8)
        class_tokens[np.arange(len(labels)), labels] = 1
        class_tokens = np.concatenate((
            class_tokens,
            -np.ones((self.max_d_elems - len(self.predictions),8))
        ), axis=0)
        return class_tokens

    @property
    def essay_id(self):
        return self.essay.essay_id

    @property
    def num_essay_words(self):
        return min(len(self.essay.words), self.max_words)

    @property
    def state(self):
        state = {
            'seg_tokens': self.seg_tokens,
            'class_tokens': self.class_tokens,
            'ner_probs': self.ner_probs
        }
        return state

    def current_state_value(self, correct_preds=False):
        if correct_preds:
            labels = self.essay.get_labels(self.predictions)
        else:
            labels = [pred.label for pred in self.predictions]
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
        if self.done:
            info.update({'Score': self.current_state_value()})
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

@register_env
class AssignmentEnv(SegmentationEnv):
    def __init__(self, essay_dataset, env_args) -> None:
        super().__init__(essay_dataset, env_args)
        self.actions = {
            0: partial(self._assign, 'None'),
            1: partial(self._assign, 'Lead'),
            2: partial(self._assign, 'Position'),
            3: partial(self._assign, 'Claim'),
            4: partial(self._assign, 'Counterclaim'),
            5: partial(self._assign, 'Rebuttal'),
            6: partial(self._assign, 'Evidence'),
            7: partial(self._assign, 'Concluding Statement'),
            8: self._merge,
        }
        self.segmented_ner_probs = torch.zeros((1,self.max_d_elems, 10))
        self.action_space = spaces.Discrete(max(self.actions.keys()))
        self.observation_space =  spaces.Box(-1,self.max_words,
                                             shape=(self.max_d_elems, 18))

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.segmented_ner_probs = segment_ner_probs(self.ner_probs)
        return self.state

    @property
    def state(self):
        state = np.concatenate((
            self.segmented_ner_probs.squeeze(0).numpy(),
            self.class_tokens
        ), axis=-1)
        state[:,9] /= 128
        return state

    def step(self, action):
        if self.done:
            raise RuntimeError('Environment is done, must be reset')
        if action not in self.actions:
            raise ValueError('Action not in available actions')
        else:
            func = self.actions[action]
            reward = func()
        info = {}
        if self.done:
            info.update({'Score': self.current_state_value()})
        else:
            info.update({'Score': None})
        return self.state, reward, self.done, info

    @property
    def segment_lens(self):
        seg_lens = self.segmented_ner_probs[0,:,-1].tolist()
        seg_lens = [seg_len for seg_len in seg_lens if seg_len != -1]
        return seg_lens

    @property
    def num_segments(self):
        return len(self.segment_lens)

    @property
    def cur_seg_idx(self):
        return len(self.predictions)

    @property
    def word_idx(self):
        return sum(self.segment_lens[:self.cur_seg_idx])

    def _merge(self):
        if (self.cur_seg_idx + 1) >= len(self.segment_lens):
            return -0.05
        init_value = self.current_state_value()
        pred_len = self.segment_lens[self.cur_seg_idx]
        start = self.word_idx
        stop = start + pred_len - 1
        pred = Prediction(start, stop, 0, self.essay_id)
        self.predictions.append(pred)
        potential_rewards = []
        for argument_name in argument_names:
            self.predictions[-1].label = argument_types[argument_name]
            potential_rewards.append(
                self.current_state_value() - init_value
            )
        bonus = max(potential_rewards) == 0
        self.predictions = self.predictions[:-1]
        
        init_value = self.current_state_value(correct_preds=True)
        cur_len = self.segment_lens[self.cur_seg_idx]
        next_len = self.segment_lens[self.cur_seg_idx + 1]
        total_len = cur_len + next_len
        merge_seg_data = self.segmented_ner_probs[:,self.cur_seg_idx:(self.cur_seg_idx+2),:]
        start_prob = merge_seg_data[0,0,0]
        prev_seg_data = self.segmented_ner_probs[:,:self.cur_seg_idx,:]
        next_seg_data = self.segmented_ner_probs[:,(self.cur_seg_idx+2):,:]
        merge_seg_data = (cur_len * merge_seg_data[:,0:1,:]
                          + next_len * merge_seg_data[:,1:2,:]) / total_len
        merge_seg_data[0,0,0] = start_prob
        merge_seg_data[0,0,-1] = total_len
        pad = -torch.ones((1,1,10))
        self.segmented_ner_probs = torch.cat((prev_seg_data, merge_seg_data,
                                              next_seg_data, pad), dim=1)
        reward = self.current_state_value(correct_preds=True)
        if bonus and reward == 0:
            reward = 0.05
        return reward

    def _assign(self, label):
        init_value = self.current_state_value()
        label = argument_types[label]
        pred_len = self.segment_lens[self.cur_seg_idx]
        start = self.word_idx
        stop = start + pred_len - 1
        pred = Prediction(start, start + pred_len - 1, label, self.essay_id)
        self.predictions.append(pred)
        if (stop + 1)>= (self.num_essay_words - 1):
            self.done = True
        reward = self.current_state_value() - init_value
        return reward

    @property
    def num_essay_words(self):
        return min(sum(self.segment_lens), self.max_words)

