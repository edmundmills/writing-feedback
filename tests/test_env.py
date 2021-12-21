from pathlib import Path

import pandas as pd
import pytest

from env import *

def test_reset():
    env = AssigmentEnv()
    env.reset()
    assert(env.done == False)
    assert(env.reward == 0)
    assert(isinstance(env.essay_id, str))
    assert(isinstance(env.essay_text, str))
    assert(isinstance(env.essay_path, Path))
    assert(isinstance(env.essay_labels, pd.DataFrame))
    assert(isinstance(env.sentences, list))
    position = torch.zeros(env.max_sentences)
    position[0] = 1
    assert(torch.equal(env.position, position))

class TestStep:
    def test_env_done(self):
        env = AssigmentEnv()
        env.done = True
        with pytest.raises(RuntimeError):
            env.step(0)
    
    def test_action_not_in_actions(self):
        env = AssigmentEnv()
        env.reset()
        with pytest.raises(ValueError):
            env.step(-1)

    def test_up_valid(self):
        env = AssigmentEnv()
        env.reset()
        expected_position = torch.zeros(env.max_sentences)
        expected_position[1] = 1
        (position, _, _), reward, done = env.step(0)
        assert(torch.equal(position, expected_position))

    def test_up_invalid(self):
        env = AssigmentEnv()
        env.reset()
        env._position = env.max_sentences - 1
        expected_position = torch.zeros(env.max_sentences)
        expected_position[-1] = 1
        (position, _, _), reward, done = env.step(0)
        assert(torch.equal(position, expected_position))

    def test_down_valid(self):
        env = AssigmentEnv()
        env.reset()
        env._position = 2
        expected_position = torch.zeros(env.max_sentences)
        expected_position[1] = 1
        (position, _, _), reward, done = env.step(1)
        assert(torch.equal(position, expected_position))

    def test_down_invalid(self):
        env = AssigmentEnv()
        env.reset()
        expected_position = torch.zeros(env.max_sentences)
        expected_position[0] = 1
        (position, _, _), reward, done = env.step(1)
        assert(torch.equal(position, expected_position))
        
    def test_end(self):
        env = AssigmentEnv()
        env.reset()
        env.step(12)
        assert(env.done == True)
