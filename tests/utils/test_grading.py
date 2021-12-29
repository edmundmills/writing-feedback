import pytest

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


class TestGrade:
    def test_single_prediction(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        predictions = [
            {
                'id': 0,
                'class': arg_class,
                'predictionstring': predictionstring,
            },
        ]
        metrics = essay.grade(predictions)
        assert(metrics['f_score'] == 1 / len(essay.labels))
        assert(metrics['true_positives'] == 1)
        assert(metrics['false_positives'] == 0)
        assert(metrics['false_negatives'] == len(essay.labels) - 1)

    def test_two_predictions(self, essay):
        arg_class1 = essay.labels.iloc[0]['discourse_type']
        predictionstring1 = essay.labels.iloc[0]['predictionstring']
        arg_class2 = essay.labels.iloc[1]['discourse_type']
        predictionstring2 = essay.labels.iloc[1]['predictionstring']
        predictions = [
            {
                'id': 0,
                'class': arg_class1,
                'predictionstring': predictionstring1,
            },
            {
                'id': 0,
                'class': arg_class2,
                'predictionstring': predictionstring2,
            },
        ]
        metrics = essay.grade(predictions)
        assert(metrics['f_score'] == 2 / len(essay.labels))
        assert(metrics['true_positives'] == 2)
        assert(metrics['false_positives'] == 0)
        assert(metrics['false_negatives'] == len(essay.labels) - 2)

    def test_mixed_predictions(self, essay):
        arg_class1 = essay.labels.iloc[0]['discourse_type']
        predictionstring1 = essay.labels.iloc[0]['predictionstring']
        arg_class2 = essay.labels.iloc[1]['discourse_type']
        predictionstring2 = essay.labels.iloc[1]['predictionstring']
        predictions = [
            {
                'id': 0,
                'class': arg_class1,
                'predictionstring': predictionstring1,
            },
            {
                'id': 0,
                'class': arg_class2,
                'predictionstring': predictionstring2,
            },
            {
                'id': 0,
                'class': arg_class1,
                'predictionstring': predictionstring2,
            },
        ]
        metrics = essay.grade(predictions)
        assert(metrics['f_score'] == 2 / (len(essay.labels) + 1))
        assert(metrics['true_positives'] == 2)
        assert(metrics['false_positives'] == 1)
        assert(metrics['false_negatives'] == len(essay.labels) - 2)

    def test_all_wrong(self, essay):
        predictions = [
            {
                'id': 0,
                'class': 'dsfg',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
            {
                'id': 0,
                'class': 'sdfg',
                'predictionstring': '45 46 47 48 49 50 51 52 53 54 55 56 57 58 59',
            },
            {
                'id': 0,
                'class': 'sdfgds',
                'predictionstring': '60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75',
            },
        ]
        metrics = essay.grade(predictions)
        assert(metrics['f_score'] == 0)
        assert(metrics['true_positives'] == 0)
        assert(metrics['false_positives'] == 3)
        assert(metrics['false_negatives'] == len(essay.labels))
 
    def test_duplicate_right(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        predictions = [
            {
                'id': 0,
                'class': arg_class,
                'predictionstring': predictionstring,
            },
            {
                'id': 0,
                'class': arg_class,
                'predictionstring': predictionstring,
            },
        ]
        metrics = essay.grade(predictions)
        assert(metrics['f_score'] == 1 / (len(essay.labels) + 1))
        assert(metrics['true_positives'] == 1)
        assert(metrics['false_positives'] == 1)
        assert(metrics['false_negatives'] == len(essay.labels) - 1)

