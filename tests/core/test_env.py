from stable_baselines3.common.env_checker import check_env

from core.env import *


class TestSegmentationEnv:
    def test_make(self, dataset_with_ner_probs, base_args):
        env = SegmentationEnv.make(2, dataset_with_ner_probs, base_args)
        assert(len(env.get_attr('max_words')) == 2)
        assert(sum(len(ds) for ds in env.get_attr('dataset')) == len(dataset_with_ner_probs))

class TestJoinEnv:
    def test_init(self, dataset_with_ner_probs, join_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, join_args)
        check_env(env)

    def test_reset(self, join_env):
        join_env.reset()
        assert(join_env.done == False)
