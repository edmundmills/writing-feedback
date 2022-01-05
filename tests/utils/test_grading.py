import pytest
import torch 

from core.constants import argument_names
from utils.grading import *

class TestPredictionString:
    def test_normal(self):
        predictionstring = prediction_string(0, 5)
        assert(predictionstring == '0 1 2 3 4 5')

    def test_same(self):
        predictionstring = prediction_string(0, 0)
        assert(predictionstring == '0')

class TestIsMatch:
    def test_exact_match(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        prediction = {
            'id': 0,
            'class': arg_class,
            'predictionstring': predictionstring,
        }
        assert(ismatch(prediction, essay.labels.iloc[0]))

    def test_short_match(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        prediction = {
            'id': 0,
            'class': arg_class,
            'predictionstring': predictionstring[:int(len(predictionstring)*.75)],
        }
        assert(ismatch(prediction, essay.labels.iloc[0]))

    def test_long_match(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        predictionstring2 = essay.labels.iloc[0]['predictionstring']
        prediction = {
            'id': 0,
            'class': arg_class,
            'predictionstring': predictionstring + predictionstring2[:int(len(predictionstring)*.25)],
        }
        assert(ismatch(prediction, essay.labels.iloc[0]))

    def test_short_nonmatch(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        prediction = {
            'id': 0,
            'class': arg_class,
            'predictionstring': predictionstring[:int(len(predictionstring)*.25)],
        }
        assert(not ismatch(prediction, essay.labels.iloc[0]))

    def test_long_nonmatch(self, essay, fix_seed):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        predictionstring2 = essay.labels.iloc[1]['predictionstring']
        predictionstring3 = essay.labels.iloc[3]['predictionstring']
        submitted_prediction_string = predictionstring + predictionstring2 + predictionstring3 + '1000 1001 1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021'
        print(len(predictionstring), len(submitted_prediction_string))
        prediction = {
            'id': 0,
            'class': arg_class,
            'predictionstring':  submitted_prediction_string,
        }
        assert(not ismatch(prediction, essay.labels.iloc[0]))

    def test_other_label(self, essay):
        arg_class = essay.labels.iloc[1]['discourse_type']
        predictionstring = essay.labels.iloc[1]['predictionstring']
        prediction = {
                'id': 0,
                'class': arg_class,
                'predictionstring': predictionstring,
        }
        assert(ismatch(prediction, essay.labels.iloc[1]))

class TestToPrediction:
    def test_to_predictions(self, fix_seed, pstrings, essay):
        logits = torch.FloatTensor(len(pstrings), len(argument_names))
        predictions = to_predictions(pstrings, logits, 1)
        print(predictions)
        grade = essay.grade(predictions)
        print(grade)
        assert(isinstance(predictions, list))
        assert(isinstance(predictions[0], dict))

class TestPstringsToTokens:
    def test_valid(self, pstrings):
        length = 50
        tokens = pstrings_to_tokens(pstrings, length)
        print(tokens)
        assert(len(tokens) == length)
        assert(set(tokens) == set(['MASK', 'CONT', 'START']))

class TestGetDiscourseElements:
    def test_valid(self, essay):
        pstrings = essay.pstrings
        text = essay.text
        print(get_discourse_elements(text, pstrings))
        print(essay.d_elems_text)
        for d_elem_a, d_elem_b in zip(essay.d_elems_text, get_discourse_elements(text, pstrings)):
            assert(d_elem_a.split() == d_elem_b.split())
