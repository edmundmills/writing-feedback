import pytest

from stable_baselines3.common.env_checker import check_env

from core.env import *
from core.dataset import Essay


class TestPstringsToTokens:
    def test_valid(self, prediction):
        length = 50
        tokens = to_tokens([prediction], length)
        print(tokens)
        assert(len(tokens) == length)
        assert(set(tokens) == set((-1, 0, 1)))


class TestSegmentationEnv:
    def test_init(self, seg_tokenizer, dataset, seg_args):
        env = SegmentationEnv(dataset, seg_tokenizer, None, seg_args)
        assert(isinstance(env, SegmentationEnv))
        check_env(env)

    def test_reset(self, seg_env):
        state = seg_env.reset()
        assert(isinstance(state, dict))
        assert(not seg_env.done)
    
    def test_act(self, seg_env):
        seg_env.reset()
        state, reward, done, info = seg_env.step(1)
        assert(isinstance(state, dict))
        assert(isinstance(reward, float))
        assert(not done)


class TestAssignmentEnv:
    def test_reset(self, env):
        env.reset()
        assert(env.done == False)
        assert(env.reward == 0)
        assert(isinstance(env.essay, Essay))
        assert(isinstance(env.sentences, list))
        position = torch.zeros(env.max_sentences)
        position[0] = 1
        assert(torch.equal(env.position, position))

    def test_env_done(self, env):
        env.done = True
        with pytest.raises(RuntimeError):
            env.step(0)
    
    def test_action_not_in_actions(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step(-1)

    def test_up_valid(self, env):
        env.reset()
        expected_position = torch.zeros(env.max_sentences)
        expected_position[1] = 1
        (position, _, _), reward, done = env.step(0)
        assert(torch.equal(position, expected_position))

    def test_up_invalid(self, env):
        env.reset()
        env._position = env.max_sentences - 1
        expected_position = torch.zeros(env.max_sentences)
        expected_position[-1] = 1
        (position, _, _), reward, done = env.step(0)
        assert(torch.equal(position, expected_position))

    def test_down_valid(self, env):
        env.reset()
        env._position = 2
        expected_position = torch.zeros(env.max_sentences)
        expected_position[1] = 1
        (position, _, _), reward, done = env.step(1)
        assert(torch.equal(position, expected_position))

    def test_down_invalid(self, env):
        env.reset()
        expected_position = torch.zeros(env.max_sentences)
        expected_position[0] = 1
        (position, _, _), reward, done = env.step(1)
        assert(torch.equal(position, expected_position))
        
    def test_end(self, env):
        env.reset()
        env.step(13)
        assert(env.done == True)
