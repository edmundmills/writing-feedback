from dataset import ArgumentDataset
from utils.grading import *

class TestIsMatch:
    def test_exact_match(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        prediction = {
            'id': 0,
            'class': 'Lead',
            'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44',
        }
        assert(ismatch(prediction, essay.iloc[0]))

    def test_short_match(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        prediction = {
            'id': 0,
            'class': 'Lead',
            'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31',
        }
        assert(ismatch(prediction, essay.iloc[0]))

    def test_long_match(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        prediction = {
            'id': 0,
            'class': 'Lead',
            'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52',
        }
        assert(ismatch(prediction, essay.iloc[0]))

    def test_short_nonmatch(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        prediction = {
            'id': 0,
            'class': 'Lead',
            'predictionstring': '1 2 3 4 5 6 7 8 ',
        }
        assert(not ismatch(prediction, essay.iloc[0]))

    def test_long_nonmatch(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        prediction = {
            'id': 0,
            'class': 'Lead',
            'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90',
        }
        assert(not ismatch(prediction, essay.iloc[0]))

    def test_other_label(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        prediction = {
                'id': 0,
                'class': 'Position',
                'predictionstring': '45 46 47 48 49 50 51 52 53 54 55 56 57 58 59',
        }
        assert(ismatch(prediction, essay.iloc[1]))


class TestGrade:
    def test_single_prediction(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        predictions = [
            {
                'id': 0,
                'class': 'Lead',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
        ]
        metrics = grade(predictions, essay)
        assert(metrics['f_score'] == .1)
        assert(metrics['true_positives'] == 1)
        assert(metrics['false_positives'] == 0)
        assert(metrics['false_negatives'] == 9)

    def test_two_predictions(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        predictions = [
            {
                'id': 0,
                'class': 'Lead',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
            {
                'id': 0,
                'class': 'Position',
                'predictionstring': '45 46 47 48 49 50 51 52 53 54 55 56 57 58 59',
            },
        ]
        metrics = grade(predictions, essay)
        assert(metrics['f_score'] == .2)
        assert(metrics['true_positives'] == 2)
        assert(metrics['false_positives'] == 0)
        assert(metrics['false_negatives'] == 8)

    def test_mixed_predictions(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        predictions = [
            {
                'id': 0,
                'class': 'Lead',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
            {
                'id': 0,
                'class': 'Position',
                'predictionstring': '45 46 47 48 49 50 51 52 53 54 55 56 57 58 59',
            },
            {
                'id': 0,
                'class': 'Concluding Statement',
                'predictionstring': '60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75',
            },
        ]
        metrics = grade(predictions, essay)
        assert(metrics['f_score'] == 0.18181818181818182)
        assert(metrics['true_positives'] == 2)
        assert(metrics['false_positives'] == 1)
        assert(metrics['false_negatives'] == 8)

    def test_all_wrong(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        predictions = [
            {
                'id': 0,
                'class': 'Position',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
            {
                'id': 0,
                'class': 'Lead',
                'predictionstring': '45 46 47 48 49 50 51 52 53 54 55 56 57 58 59',
            },
            {
                'id': 0,
                'class': 'Concluding Statement',
                'predictionstring': '60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75',
            },
        ]
        metrics = grade(predictions, essay)
        assert(metrics['f_score'] == 0)
        assert(metrics['true_positives'] == 0)
        assert(metrics['false_positives'] == 3)
        assert(metrics['false_negatives'] == 10)
 
    def test_duplicate_right(self):
        dataset = ArgumentDataset()
        essay = next(dataset.essays())
        predictions = [
            {
                'id': 0,
                'class': 'Lead',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
            {
                'id': 0,
                'class': 'Lead',
                'predictionstring': '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49',
            },
        ]
        metrics = grade(predictions, essay)
        assert(metrics['f_score'] == 0.09090909090909091)
        assert(metrics['true_positives'] == 1)
        assert(metrics['false_positives'] == 1)
        assert(metrics['false_negatives'] == 9)

