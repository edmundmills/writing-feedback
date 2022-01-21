from core.ner import *


class TestEncode:
    def test_valid(self, ner_args, essay):
        tokenizer = NERTokenizer(ner_args)
        tokenized = tokenizer.encode(essay.text)
        assert(tokenized['input_ids'].size() == (1, 1024))
        assert(tokenized['attention_mask'].size() == (1, 1024))
        assert(len(tokenized['word_ids']) == 1024)
        assert(tokenized['word_id_tensor'].size() == (1,1024))

class TestGetPredLabels:
    def test_valid(self, prediction):
        length = 50
        tokens = get_pred_labels([prediction], length)
        start_tokens = tokens[0]
        class_tokens = tokens[1]
        assert(tokens.shape == (2,length))
        assert(set(start_tokens) == set((-1, 0, 1)))
        assert(min(class_tokens) == -1)
        assert(max(class_tokens) <= 7)



class TestNERModel:
    def test_forward_ner_seg_only(self, encoded_essay, ner_args):
        ner_args.segmentation_only = True
        model = NERModel(ner_args)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        output = model(encoded_text.to(model.device),
                       attention_mask.to(model.device))
        assert(output.size() == (1, ner_args.essay_max_tokens, 2))

    def test_forward_ner(self, encoded_essay, ner_args):
        ner_args.segmentation_only = False
        model = NERModel(ner_args)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        output = model(encoded_text.to(model.device),
                       attention_mask.to(model.device))
        assert(output.size() == (1, ner_args.essay_max_tokens, 10))

    def test_collate_word_idxs(self, ner_args):
        model = NERModel(ner_args)
        probs = torch.FloatTensor([0, .1, .2, .3, .4, .5, 0]).unsqueeze(0).unsqueeze(-1)
        word_ids = torch.LongTensor([-1, 0, 1, 1, 2, 2, -1]).unsqueeze(0)
        new_probs = model.collate_word_idxs(probs, word_ids)
        print(new_probs)
        assert(torch.equal(new_probs,
                           torch.FloatTensor([0, .1, .2, .4, 0, 0, 0]).unsqueeze(0).unsqueeze(-1)))
        probs = torch.FloatTensor([[0, .1, .2], [0, .4, .5]]).unsqueeze(-1)
        word_ids = torch.LongTensor([[-1, 0, 0], [-1, 0, 0]])
        new_probs = model.collate_word_idxs(probs, word_ids)
        assert(torch.equal(new_probs, torch.FloatTensor([[0, .1, 0], [0, .4, 0]]).unsqueeze(-1)))
        probs = torch.FloatTensor([[0, .1, .2], [0, .4, .5]]).unsqueeze(-1)
        word_ids = torch.LongTensor([[-1, 0, 1], [-1, 0, 0]])
        new_probs = model.collate_word_idxs(probs, word_ids)
        assert(torch.equal(new_probs, torch.FloatTensor([[0, .1, .2], [0, .4, 0]]).unsqueeze(-1)))
        probs = torch.FloatTensor([[0, .1, .2], [0, .4, .5]]).unsqueeze(-1)
        word_ids = torch.LongTensor([[-1, 0, 0]])
        new_probs = model.collate_word_idxs(probs, word_ids)
        assert(torch.equal(new_probs, torch.FloatTensor([[0, .1, 0], [0, .4, 0]]).unsqueeze(-1)))

    def test_inference_seg_only(self, encoded_essay, base_args):
        base_args.ner.segmentation_only = True
        model = NERModel(base_args.ner)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        word_ids = encoded_essay['word_id_tensor']
        output = model.inference(encoded_text, attention_mask, word_ids)
        assert(output.size() == (1, base_args.ner.essay_max_tokens, 1))

    def test_inference(self, encoded_essay, base_args):
        base_args.ner.segmentation_only = False
        model = NERModel(base_args.ner)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        word_ids = encoded_essay['word_id_tensor']
        output = model.inference(encoded_text, attention_mask, word_ids)
        assert(output.size() == (1, base_args.ner.essay_max_tokens, 9))

    def test_make_ner_dataset_seg_only(self, fix_seed, dataset, ner_tokenizer):
        essay_feedback_dataset = ner_tokenizer.make_ner_dataset(dataset, seg_only=True)
        # input_ids
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (1024,))
        # attention_masks
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (1024,))
        #labels
        assert(isinstance(essay_feedback_dataset[0][2], torch.Tensor))
        assert(essay_feedback_dataset[0][2].size() == (1024,))
        assert(essay_feedback_dataset[0][2].max().item() == 1)
        assert(essay_feedback_dataset[0][2].min().item() == -1)

    def test_make_ner_dataset(self, fix_seed, dataset, ner_tokenizer):
        essay_feedback_dataset = ner_tokenizer.make_ner_dataset(dataset)
        # input_ids
        assert(isinstance(essay_feedback_dataset[0][0], torch.Tensor))
        assert(essay_feedback_dataset[0][0].size() == (1024,))
        # attention_masks
        assert(isinstance(essay_feedback_dataset[0][1], torch.Tensor))
        assert(essay_feedback_dataset[0][1].size() == (1024,))
        #labels
        assert(isinstance(essay_feedback_dataset[0][2], torch.Tensor))
        assert(essay_feedback_dataset[0][2].size() == (1024, 2))
        assert(essay_feedback_dataset[0][2].max().item() <= 15)
        assert(essay_feedback_dataset[0][2].min().item() == -1)
