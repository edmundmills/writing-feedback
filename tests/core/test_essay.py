from copy import copy

from core.essay import Prediction

from utils.constants import de_num_to_type, de_type_to_num

class TestPrediction:
    def test_formatted(self, prediction):
        formatted = prediction.formatted()
        assert(formatted['id'] == prediction.essay_id)
        assert(formatted['predictionstring'] == prediction.pstring)
        assert(formatted['class'] == prediction.argument_name)

    def test_word_idxs(self, prediction):
        assert(len(prediction.word_idxs) == (prediction.stop - prediction.start + 1))

class TestEssay:
    def test_words(self, essay):
        words = essay.words
        assert(isinstance(words, list))
        assert(isinstance(words[0], str))

    def test_all_d_elems(self, essay):
        arguments = essay._all_d_elems()
        assert(isinstance(arguments, list))
        assert(isinstance(arguments[0], tuple))
        assert(isinstance(arguments[0][0], str))
        assert(isinstance(arguments[0][1], str))
        assert(len(arguments) >= len(essay.labels))
        print(arguments)
        assert(sum(len(argument.split()) for argument, _, _ in arguments) == len(essay.words))

    def test_d_elems_text(self, essay):
        assert(isinstance(essay.d_elems_text, list))
        assert(essay.d_elems_text[0] == essay.labels.iloc[0].loc['discourse_text'])

    def test_pstrings(self, essay):
        assert(isinstance(essay.pstrings, list))
        assert(essay.pstrings[0] == essay.labels.iloc[0].loc['predictionstring'])

    def test_random_pstrings(self, essay):
        pstrings = essay.random_pstrings()
        assert(isinstance(pstrings, list))
        assert(isinstance(pstrings[0], str))
        assert(all([isinstance(int(num), int) for num in pstrings[0].split()]))
        nums = [int(num) for pstring in pstrings for num in pstring.split()]
        nums = set(nums)
        assert(nums == set(range(len(essay.words))))

    def test_random_pstrings_with_max(self, essay):
        pstrings = essay.random_pstrings(max_d_elems=2)
        assert(len(pstrings) <= 2)
       
    def test_correct_predictions(self, essay):
        preds = essay.correct_predictions
        assert(isinstance(preds, list))
        assert(isinstance(preds[0], Prediction))
        assert(len(preds) == len(essay._all_d_elems()))
        assert(preds[0].start == 0)
        assert(preds[-1].stop == len(essay.words) - 1)
  

class TestGetLabelsForSegments:
    def test_correct_preds(self, essay):
        segment_lens = [len(pred) for pred in essay.correct_predictions]
        segment_preds = essay.get_labels_for_segments(segment_lens)
        print(segment_preds)
        assert(segment_preds == [(len(pred), pred.label) for pred in essay.correct_predictions])


class TestSegLabelsToPreds:
    def test_correct_preds(self, essay):
        preds = essay.correct_predictions
        segment_preds = essay.get_labels_for_segments(preds)
        new_preds = essay.segment_labels_to_preds(segment_preds)
        assert(all([pred == new_pred for pred, new_pred in zip(preds, new_preds)]))


class TestNERLabels:
    def test_valid(self, prediction, essay):
        length = 50
        print(prediction)
        pred2 = copy(prediction)
        pred2.start = 20
        pred2.stop = 30
        pred2.label = 5
        tokens = essay.ner_labels(length, predictions=[prediction, pred2])
        print(tokens)
        assert(tokens.shape == (length,))
        assert(min(tokens) == -1)
        assert(max(tokens) <= 15)


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

    def test_none(self, essay):
        arg_class = essay.labels.iloc[0]['discourse_type']
        predictionstring = essay.labels.iloc[0]['predictionstring']
        predictions = [
            {
                'id': 0,
                'class': 'None',
                'predictionstring': predictionstring,
            },
        ]
        metrics = essay.grade(predictions)
        assert(metrics['f_score'] == 0)
        assert(metrics['true_positives'] == 0)
        assert(metrics['false_positives'] == 0)
        assert(metrics['false_negatives'] == len(essay.labels))

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

class TestGetLabel:
    def test_exact(self, essay):
        pstring = essay.pstrings[0]
        label = essay.get_labels(pstring)[0]
        true_label = de_type_to_num[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': de_num_to_type[label],
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
        label = essay.get_labels(pstring)[0]
        true_label = de_type_to_num[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': de_num_to_type[label],
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
        label = essay.get_labels(pstring)[0]
        true_label = de_type_to_num[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': de_num_to_type[label],
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
        label = essay.get_labels(pstring)[0]
        true_label = de_type_to_num[essay.labels.iloc[0].loc['discourse_type']]
        prediction = [{'id': essay.essay_id,
                       'class': de_num_to_type[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == 0)
        assert(grading_data['true_positives'] == 0)
        assert(grading_data['false_positives'] == 0)
    
    def test_too_long(self, essay):
        nums = range(len(essay.words))
        pstring = ' '.join(str(num) for num in nums)
        label = essay.get_labels(pstring)[0]
        prediction = [{'id': essay.essay_id,
                       'class': de_num_to_type[label],
                       'predictionstring': pstring}]
        grading_data = essay.grade(prediction)
        assert(isinstance(label, int))
        assert(label == 0)
        assert(grading_data['true_positives'] == 0)
        assert(grading_data['false_positives'] == 0)


class TestGetLabels:
    def test_all_correct(self, essay):
        labels = essay.get_labels(essay.pstrings)
        predictions = [
            {'id': essay.essay_id,
             'class': de_num_to_type[label],
             'predictionstring': pstring}
            for pstring, label in zip(essay.pstrings, labels)
        ]
        grading_data = essay.grade(predictions)
        assert(grading_data['f_score'] == 1)

    def test_with_num_specified(self, essay):
        labels = essay.get_labels(essay.pstrings, num_d_elems=10)
        assert(len(labels) == 10)
        labels = essay.get_labels(essay.pstrings, num_d_elems=2)
        assert(len(labels) == 2)
        labels = essay.get_labels(essay.pstrings, num_d_elems=40)
        assert(len(labels) == 40)