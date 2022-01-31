import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from utils.constants import de_num_to_type, de_len_norm_factor
from core.segmenter import Segmenter
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
        self.segmenter = Segmenter(args.seg)

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
        return 0 if len(self.predictions) == 0 else (self.predictions[-1].stop + 1)

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
        self.done = False
        self.steps = 0
        segmented_ner_probs, segment_lens = self.segmenter.segment_ner_probs(self.essay.ner_probs)
        self.segment_lens = [seg_len for seg_len in segment_lens if seg_len != -1]
        self.correct_labels = [label for _len, label
                       in self.essay.get_labels_for_segments(self.segment_lens)]
        segmented_ner_probs = segmented_ner_probs.squeeze(0).numpy()
        segmented_ner_probs[:,-1] /= de_len_norm_factor
        self.segmented_ner_probs = segmented_ner_probs
        return self.state
        
    def step(self, action):
        init_value_join = self.current_state_value(correct_preds=True)
        init_value_class = self.current_state_value(correct_preds=False)
        self.update_state(action)
        value_join = self.current_state_value(correct_preds=True)
        value_class = self.current_state_value(correct_preds=False)
        reward = (value_join - init_value_join + value_class - init_value_class) / 2
        info = {}
        if self.done:
            print('##############')
            print(self.current_state_value(correct_preds=True))
            for pred in self.essay.correct_predictions:
                print(pred.start, pred.stop, pred.label)
            for pred in self.predictions:
                print(pred.start, pred.stop, pred.label)

            score = self.current_state_value()
            info.update({'Score': score})
            print(score)
        else:
            info.update({'Score': None})
        return self.state, reward, self.done, info

    def update_state(self, action):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError
    
    @property
    def predictions(self):
        raise NotImplementedError

@SegmentationEnv.register
class JoinEnv(SegmentationEnv):
    def __init__(self, essay_dataset, args) -> None:
        super().__init__(essay_dataset, args)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(self.max_d_elems, 17))
        self.action_space = spaces.Discrete(15)
        self.segmented_ner_probs = np.zeros(self.observation_space.shape)
        self.pred_labels = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.pred_labels = []
        return self.state

    @property
    def attention_mask(self):
        msk = np.zeros((self.max_d_elems, 1))
        msk[len(self.pred_labels), ...] = 1
        return msk

    @property
    def state(self):
        return np.concatenate((
            self.segmented_ner_probs,
            self.attention_mask,
        ), axis=-1)

    @property
    def predictions(self):
        labels = list(zip(self.segment_lens[:len(self.pred_labels)], self.pred_labels))
        predictions = self.essay.segment_labels_to_preds(labels)
        return predictions

    def update_state(self, action):
        self.pred_labels.append(int(action))
        if len(self.pred_labels) >= self.num_segments:
            self.done = True

