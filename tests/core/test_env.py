import pytest

from stable_baselines3.common.env_checker import check_env
from torch import div

from core.env import *
from core.dataset import Essay
from core.segmentation import make_agent




class TestDividerEnv:
    def test_init(self, ner_tokenizer, dataset, d_elem_tokenizer, base_args):
        env = DividerEnv(dataset, ner_tokenizer, d_elem_tokenizer, base_args.env)
        assert(isinstance(env, DividerEnv))
        check_env(env)

    def test_reset(self, divider_env):
        state = divider_env.reset()
        assert(isinstance(state, dict))
        assert(not divider_env.done)
        assert(len(divider_env.predictions) == divider_env.max_d_elems)
        for idx in range(divider_env.max_d_elems - 1):
            pred1 = divider_env.predictions[idx]
            pred2 = divider_env.predictions[idx + 1]
            assert(pred1.stop + 1 == pred2.start)
    
    def test_act(self, divider_env):
        divider_env.reset()
        action = np.zeros(32)
        action[2] = 2
        pred1 = divider_env.predictions[2]
        pred2 = divider_env.predictions[3]
        stop, start = pred1.stop, pred2.start
        state, reward, done, info = divider_env.step(action)
        pred1 = divider_env.predictions[2]
        pred2 = divider_env.predictions[3]
        assert(isinstance(state, dict))
        assert(isinstance(reward, float))
        assert(stop + 1 == pred1.stop)
        assert(start + 1 == pred2.start)
        assert(not done)

    # def test_act_done(self, divider_env):
    #     divider_env.reset()
    #     action = np.zeros(32)
    #     action[-1] = 1
    #     state, reward, done, info = divider_env.step(action)
    #     assert(done)

    def test_make_vec(self, base_args, ner_tokenizer, dataset, d_elem_tokenizer):
        env = DividerEnv.make_vec(2, dataset, ner_tokenizer, d_elem_tokenizer, base_args.env)
        assert(len(env.get_attr('done')) == 2)
        assert(sum(len(ds) for ds in env.get_attr('dataset')) == len(dataset))
        make_agent(base_args, env)


class TestWordwiseEnv:
    def test_init(self, ner_tokenizer, dataset, d_elem_tokenizer, base_args):
        env = WordwiseEnv(dataset, ner_tokenizer, d_elem_tokenizer, base_args.env)
        assert(isinstance(env, WordwiseEnv))
        check_env(env)

    def test_reset(self, word_env):
        state = word_env.reset()
        assert(isinstance(state, dict))
        assert(not word_env.done)
    
    def test_act(self, word_env):
        word_env.reset()
        action = np.zeros(1 + 1024 + 32)
        state, reward, done, info = word_env.step(action)
        assert(isinstance(state, dict))
        assert(isinstance(reward, float))
        assert(not done)

    def test_make_vec(self, base_args, ner_tokenizer, dataset, d_elem_tokenizer):
        env = WordwiseEnv.make_vec(2, dataset, ner_tokenizer, d_elem_tokenizer, base_args.env)
        assert(len(env.get_attr('done')) == 2)
        assert(sum(len(ds) for ds in env.get_attr('dataset')) == len(dataset))
        make_agent(base_args, env)


class TestSequencewiseEnv:
    def test_init(self, ner_tokenizer, dataset, d_elem_tokenizer, base_args):
        env = SequencewiseEnv(dataset, ner_tokenizer, d_elem_tokenizer, base_args.env)
        assert(isinstance(env, SequencewiseEnv))
        check_env(env)

    def test_reset(self, seq_env):
        state = seq_env.reset()
        assert(isinstance(state, dict))
        assert(not seq_env.done)
    
    def test_act(self, seq_env):
        seq_env.reset()
        state, reward, done, info = seq_env.step(1)
        assert(isinstance(state, dict))
        assert(isinstance(reward, float))
        assert(not done)

    def test_make_vec(self, ner_tokenizer, dataset, d_elem_tokenizer, base_args):
        env = SequencewiseEnv.make_vec(2, dataset, ner_tokenizer, d_elem_tokenizer, base_args.env)
        assert(len(env.get_attr('done')) == 2)
        assert(sum(len(ds) for ds in env.get_attr('dataset')) == len(dataset))
        make_agent(base_args, env)


class TestAssignmentEnv:
    def test_reset(self, assign_env):
        assign_env.reset()
        assert(assign_env.done == False)
        assert(assign_env.reward == 0)
        assert(isinstance(assign_env.essay, Essay))
        assert(isinstance(assign_env.sentences, list))
        position = torch.zeros(assign_env.max_sentences)
        position[0] = 1
        assert(torch.equal(assign_env.position, position))

    def test_env_done(self, assign_env):
        assign_env.done = True
        with pytest.raises(RuntimeError):
            assign_env.step(0)
    
    def test_action_not_in_actions(self, assign_env):
        assign_env.reset()
        with pytest.raises(ValueError):
            assign_env.step(-1)

    def test_up_valid(self, assign_env):
        assign_env.reset()
        expected_position = torch.zeros(assign_env.max_sentences)
        expected_position[1] = 1
        (position, _, _), reward, done = assign_env.step(0)
        assert(torch.equal(position, expected_position))

    def test_up_invalid(self, assign_env):
        assign_env.reset()
        assign_env._position = assign_env.max_sentences - 1
        expected_position = torch.zeros(assign_env.max_sentences)
        expected_position[-1] = 1
        (position, _, _), reward, done = assign_env.step(0)
        assert(torch.equal(position, expected_position))

    def test_down_valid(self, assign_env):
        assign_env.reset()
        assign_env._position = 2
        expected_position = torch.zeros(assign_env.max_sentences)
        expected_position[1] = 1
        (position, _, _), reward, done = assign_env.step(1)
        assert(torch.equal(position, expected_position))

    def test_down_invalid(self, assign_env):
        assign_env.reset()
        expected_position = torch.zeros(assign_env.max_sentences)
        expected_position[0] = 1
        (position, _, _), reward, done = assign_env.step(1)
        assert(torch.equal(position, expected_position))
        
    def test_end(self, assign_env):
        assign_env.reset()
        assign_env.step(13)
        assert(assign_env.done == True)
