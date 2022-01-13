from core.essay import Prediction


class TestPrediction:
    def test_formatted(self, prediction):
        formatted = prediction.formatted()
        assert(formatted['id'] == prediction.essay_id)
        assert(formatted['predictionstring'] == prediction.pstring)
        assert(formatted['class'] == prediction.argument_name)

    def test_word_idxs(self, prediction):
        assert(len(prediction.word_idxs) == (prediction.stop - prediction.start + 1))

class TestEssay:
    def test_polarity_pairs(self, essay):
        pairs, labels = essay.polarity_pairs()
        assert(isinstance(pairs, list))
        assert(isinstance(labels, list))
        assert(len(pairs) == len(labels))
        assert(min(labels) >= -1)
        assert(max(labels) <= 1)

    def test_words(self, essay):
        words = essay.words
        assert(isinstance(words, list))
        assert(isinstance(words[0], str))

    def test_all_arguments(self, essay):
        arguments = essay.all_arguments()
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
        assert(len(preds) == len(essay.all_arguments()))
        assert(preds[0].start == 0)
        assert(preds[-1].stop == len(essay.words) - 1)


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