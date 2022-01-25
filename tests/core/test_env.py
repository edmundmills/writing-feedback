# import pytest

# from stable_baselines3.common.env_checker import check_env

# from core.env import *
# from core.rl import make_agent


# class TestSegmentationEnv:
#     def test_make(self, dataset_with_ner_probs, base_args):
#         env = SegmentationEnv.make(2, dataset_with_ner_probs, base_args.env)
#         assert(len(env.get_attr('max_words')) == 2)
#         assert(sum(len(ds) for ds in env.get_attr('dataset')) == len(dataset_with_ner_probs))
#         make_agent(base_args, env)


# class TestAssignmentEnv:
#     def test_init(self, dataset_with_ner_probs, assign_args):
#         env = SegmentationEnv.make(1, dataset_with_ner_probs, assign_args)
#         check_env(env)

#     def test_reset(self, assign_env):
#         assign_env.reset()
#         assert(assign_env.done == False)

#     def test_merge(self, assign_env):
#         assign_env.reset()
#         num_segments = assign_env.num_segments
#         prev_cur_segment = assign_env.segmented_ner_probs[0,0,:]
#         prev_next_segment = assign_env.segmented_ner_probs[0,1,:]
#         assign_env.step(8)
#         cur_segment = assign_env.segmented_ner_probs[0,0,:]
#         assert(assign_env.num_segments == num_segments - 1)
#         assert(prev_cur_segment[0] == cur_segment[0])
#         assert(prev_cur_segment[-1] + prev_next_segment[-1] == cur_segment[-1])
#         assert(all((prev_cur_segment[1:-1] != cur_segment[1:-1]).tolist()))

#     def test_assign(self, assign_env):
#         assign_env.reset()
#         assert(len(assign_env.predictions) == 0)
#         action = 0
#         assign_env.step(action)
#         assert(len(assign_env.predictions) == 1)
#         assert(assign_env.class_tokens[0,action] == 1)
#         assert(assign_env.predictions[0].label == action)
#         assert(len(assign_env.predictions[0]) == assign_env.segment_lens[0])