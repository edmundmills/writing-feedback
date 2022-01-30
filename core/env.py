from asyncio import proactor_events
import copy
from functools import partial

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from utils.constants import de_num_to_type, de_type_to_num
from core.essay import Prediction
from core.predicter import Predicter
from utils.render import plot_ner_output


env_classes = {}


class SegmentationEnv(gym.Env):
    @staticmethod
    def register(cls):
        env_classes.update({cls.__name__: cls})
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)
        return wrapper

    @staticmethod
    def make(n_envs, essay_dataset, args):
        cls = env_classes[args.env.class_name]
        if n_envs == 1:
            print(f'Making Non-Vectorized {cls.__name__} Environment')
            env = cls(essay_dataset, args)
            env = Monitor(env, filename=f'./log/test', info_keywords=('Score',))
            print('Env Created')
            return env
        print(f'Making Vectorized {cls.__name__} Environment')
        dataset_fracs = [1 / n_envs] * n_envs
        datasets = essay_dataset.split(dataset_fracs)
        def make_env(dataset):
            def _init():
                env = cls(dataset, args)
                return env
            return _init
        venv = SubprocVecEnv([make_env(ds) for ds in datasets])
        venv = VecMonitor(venv, filename=f'./log/test', info_keywords=('Score',))
        print('Vectorized env created')
        return venv


    def __init__(self, essay_dataset, args) -> None:
        super().__init__()
        self.dataset = essay_dataset
        self.max_d_elems = args.env.num_d_elems
        self.max_words = args.env.num_words
        self.essay = None
        self.grade_classifications = args.env.grade_classifications
        self.predicter = Predicter(args.predict)

    @property
    def essay_id(self):
        return self.essay.essay_id

    @property
    def num_essay_words(self):
        return min(len(self.essay.words), self.max_words)

    @property
    def num_segments(self):
        return len(self.segment_lens)

    @property
    def n_preds_made(self):
        return len(self.predictions)

    @property
    def word_idx(self):
        return sum(self.segment_lens[:self.n_preds_made])

    def current_state_value(self, correct_preds=False):
        if correct_preds:
            labels = self.essay.get_labels(self.predictions)
        else:
            labels = [pred.label for pred in self.predictions]
        predictions = [
            {'id': self.essay.essay_id,
             'class': de_num_to_type[label],
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
        self.steps = 0
        self.env_init_value = self.current_state_value()
        segmented_ner_probs, segment_lens = self.predicter.segment_ner_probs(self.essay.ner_probs)
        self.segment_lens = [seg_len for seg_len in segment_lens if seg_len != -1]
        segmented_ner_probs = segmented_ner_probs.squeeze(0).numpy()
        segmented_ner_probs[:,-1] /= 256
        self.segmented_ner_probs = segmented_ner_probs
        return self.state

    def step(self, action):
        init_value = self.current_state_value(correct_preds=True)
        self._update_state(action)
        reward = self.current_state_value(correct_preds=True) - init_value
        info = {}
        if self.done:
            info.update({'Score': self.current_state_value(correct_preds=True)})
        else:
            info.update({'Score': None})
        return self.state, reward, self.done, info

    def _update_state(self, action):
        raise NotImplementedError


@SegmentationEnv.register
class JoinEnv(SegmentationEnv):
    def __init__(self, essay_dataset, args) -> None:
        super().__init__(essay_dataset, args)
        self.predicter = Predicter(args.predict)
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=self.max_d_elems,
                                     shape=(1,)),
            'features': spaces.Box(low=-1, high=1,
                                     shape=(self.max_d_elems, 16)),
        })
        self.action_space = spaces.Discrete(2)
        self.segmented_ner_probs = np.zeros((self.max_d_elems, 16))
        self._position = None
    
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.cur_pred_seg_start = 0
        self._position = 1
        return self.state

    @property
    def position(self):
        return np.array([self._position])

    @property
    def state(self):
        state = {
            'position': self.position,
            'features': self.segmented_ner_probs
        }
        return state 
    
    def _create_cur_pred(self, action):
        pred_len = sum(self.segment_lens[self.cur_pred_seg_start:self._position])
        start = self.word_idx
        stop = self.word_idx + pred_len - 1
        self.predictions.append(
            Prediction(start, stop, 0, self.essay_id)
        )
        self.cur_pred_seg_start = self._position

    def _update_state(self, action):
        if action == 1:
            self._create_cur_pred(action)
        self._position += 1
        if self._position >= self.num_segments:
            if action != 0:
                self._create_cur_pred(action)
            self.done = True


@SegmentationEnv.register
class AssignmentEnv(SegmentationEnv):
    def __init__(self, essay_dataset, args) -> None:
        super().__init__(essay_dataset, args)
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
    def state(self):
        state = np.concatenate((
            self.segmented_ner_probs,
            self.class_tokens
        ), axis=-1)
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
        for argument_name in de_num_to_type:
            self.predictions[-1].label = de_type_to_num[argument_name]
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
        label = de_type_to_num[label]
        pred_len = self.segment_lens[self.cur_seg_idx]
        start = self.word_idx
        stop = start + pred_len - 1
        pred = Prediction(start, start + pred_len - 1, label, self.essay_id)
        self.predictions.append(pred)
        if (stop + 1)>= (self.num_essay_words - 1):
            self.done = True
        reward = self.current_state_value() - init_value
        return reward


