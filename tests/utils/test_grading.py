import pytest
import torch 

from core.constants import argument_names, argument_types
from utils.grading import *

class TestPstringsToTokens:
    def test_valid(self, prediction):
        length = 50
        tokens = to_tokens([prediction], length)
        print(tokens)
        assert(len(tokens) == length)
        assert(set(tokens) == set((-1, 0, 1)))


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

class TestGetDiscourseElements:
    def test_valid(self, essay):
        pstrings = essay.pstrings
        text = essay.text
        print(get_discourse_elements(text, pstrings))
        print(essay.d_elems_text)
        for d_elem_a, d_elem_b in zip(essay.d_elems_text, get_discourse_elements(text, pstrings)):
            assert(d_elem_a.split() == d_elem_b.split())

class TestGetLabel:
    def test_exact(self, essay):
        pstring = essay.pstrings[0]
        label = get_label(pstring, essay)
        true_label = argument_types[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': argument_names[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == true_label)
        assert(grading_data['true_positives'] == 1)
        assert(grading_data['false_positives'] == 0)
    
    def test_longer(self, essay):
        pstring = essay.pstrings[0]
        nums = [int(num) for num in pstring.split()]
        nums += list(range(max(nums)+1, max(nums) + len(nums) // 2))
        pstring = ' '.join(str(num) for num in nums)
        label = get_label(pstring, essay)
        true_label = argument_types[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': argument_names[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == true_label)
        assert(grading_data['true_positives'] == 1)
        assert(grading_data['false_positives'] == 0)

    def test_shorter(self, essay):
        pstring = essay.pstrings[0]
        nums = [int(num) for num in pstring.split()]
        nums = nums[:(len(nums)//2 + 1)]
        pstring = ' '.join(str(num) for num in nums)
        label = get_label(pstring, essay)
        true_label = argument_types[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': argument_names[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == true_label)
        assert(grading_data['true_positives'] == 1)
        assert(grading_data['false_positives'] == 0)

    def test_too_short(self, essay):
        pstring = essay.pstrings[0]
        nums = [int(num) for num in pstring.split()]
        nums = nums[:(len(nums)//2 - 1)]
        pstring = ' '.join(str(num) for num in nums)
        label = get_label(pstring, essay)
        true_label = argument_types[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': argument_names[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == 0)
        assert(grading_data['true_positives'] == 0)
        assert(grading_data['false_positives'] == 0)
    
    def test_too_long(self, essay):
        nums = range(len(essay.words))
        pstring = ' '.join(str(num) for num in nums)
        label = get_label(pstring, essay)
        prediction = [{'id': essay.essay_id,
                       'class': argument_names[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == 0)
        assert(grading_data['true_positives'] == 0)
        assert(grading_data['false_positives'] == 0)


class TestGetLabels:
    def all_correct(self, essay):
        labels = get_labels(essay.pstrings, essay)
        predictions = [
            {'id': essay.essay_id,
             'class': argument_names[label],
             'predictionstring': pstring}
            for pstring, label in zip(essay.pstrings, labels)
        ]
        grading_data = essay.grade(predictions)
        assert(grading_data['f_score'] == 1)

    def with_num_specified(self, essay):
        labels = get_labels(essay.pstrings, essay, num_d_elems=10)
        assert(len(labels) == 10)
        labels = get_labels(essay.pstrings, essay, num_d_elems=2)
        assert(len(labels) == 2)
        labels = get_labels(essay.pstrings, essay, num_d_elems=40)
        assert(len(labels) == 40)

