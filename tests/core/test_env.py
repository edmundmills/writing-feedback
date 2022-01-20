import pytest

from stable_baselines3.common.env_checker import check_env

from core.env import *
from core.dataset import Essay
from core.segmentation import make_agent


class TestSegmentationEnv:
    def test_make(self, dataset_with_ner_probs, base_args):
        env = SegmentationEnv.make(2, dataset_with_ner_probs, base_args.env)
        assert(len(env.get_attr('max_words')) == 2)
        assert(sum(len(ds) for ds in env.get_attr('dataset')) == len(dataset_with_ner_probs))
        make_agent(base_args, env)


class TestSequencewiseEnv:
    def test_init(self, dataset_with_ner_probs, seqwise_args):
        seqwise_args.continuous = False
        env = SegmentationEnv.make(1, dataset_with_ner_probs, seqwise_args)
        check_env(env)
        seqwise_args.continuous = True
        env = SegmentationEnv.make(1, dataset_with_ner_probs, seqwise_args)
        check_env(env)

    def test_reset(self, seq_env):
        state = seq_env.reset()
        assert(isinstance(state, dict))
        assert(not seq_env.done)
    
    def test_act(self, seq_env):
        seq_env.reset()
        seq_env.continuous = False
        # state, reward, done, info = seq_env.step([0,0,1,2])
        state, reward, done, info = seq_env.step(20)
        seq_env.continuous = True
        state, reward, done, info = seq_env.step(-.5)
        assert(isinstance(state, dict))
        assert(isinstance(reward, float))
        assert(not done)


class TestSplitterEnv:
    def test_init(self, dataset_with_ner_probs, splitter_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, splitter_args)
        check_env(env)

    def test_reset(self, splitter_env):
        state = splitter_env.reset()
        assert(isinstance(state, dict))
        assert(not splitter_env.done)
    
    def test_act(self, splitter_env):
        splitter_env.reset()
        init_preds = splitter_env.pred_tokens_to_preds()
        state, reward, done, info = splitter_env.step(100)
        preds = splitter_env.pred_tokens_to_preds()
        assert(len(init_preds) == 1)
        assert(len(preds) == 2)
        assert(isinstance(state, dict))
        assert(isinstance(preds[0], Prediction))
        assert(isinstance(reward, float))
        assert(not done)


class TestDividerEnv:
    def test_init(self, dataset_with_ner_probs, divider_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, divider_args)
        check_env(env)
        env = SegmentationEnv.make(1, dataset_with_ner_probs, divider_args)
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


class TestWordwiseEnv:
    def test_init(self, dataset_with_ner_probs, wordwise_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, wordwise_args)
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


class TestAssignmentEnv:
    def test_init(self, dataset_with_ner_probs, assign_args):
        env = SegmentationEnv.make(1, dataset_with_ner_probs, assign_args)
        check_env(env)

    def test_reset(self, assign_env):
        assign_env.reset()
        assert(assign_env.done == False)

    def test_merge(self, assign_env):
        assign_env.reset()
        num_segments = assign_env.num_segments
        prev_cur_segment = assign_env.segmented_ner_probs[0,0,:]
        prev_next_segment = assign_env.segmented_ner_probs[0,1,:]
        assign_env.step(8)
        cur_segment = assign_env.segmented_ner_probs[0,0,:]
        assert(assign_env.num_segments == num_segments - 1)
        assert(prev_cur_segment[0] == cur_segment[0])
        assert(prev_cur_segment[-1] + prev_next_segment[-1] == cur_segment[-1])
        assert(all((prev_cur_segment[1:-1] != cur_segment[1:-1]).tolist()))

    def test_assign(self, assign_env):
        assign_env.reset()
        assert(len(assign_env.predictions) == 0)
        action = 0
        assign_env.step(action)
        assert(len(assign_env.predictions) == 1)
        assert(assign_env.class_tokens[0,action] == 1)
        assert(assign_env.predictions[0].label == action)
        assert(len(assign_env.predictions[0]) == assign_env.segment_lens[0])