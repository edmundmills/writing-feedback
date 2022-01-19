from core.ner import *


class TestEncode:
    def test_valid(self, ner_args, essay):
        tokenizer = NERTokenizer(ner_args)
        tokenized = tokenizer.encode(essay.text)
        assert(tokenized['input_ids'].size() == (1, 1024))
        assert(tokenized['attention_mask'].size() == (1, 1024))
        assert(len(tokenized['word_ids']) == 1024)
        assert(tokenized['word_id_tensor'].size() == (1,1024))


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
        model = NERModel(base_args.ner, feature_extractor=True)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        word_ids = encoded_essay['word_id_tensor']
        x = torch.stack((encoded_text, attention_mask, word_ids), dim=1).to(model.device)
        output = model.inference(x)
        assert(output.size() == (1, base_args.ner.essay_max_tokens, 1))

    def test_inference(self, encoded_essay, base_args):
        base_args.ner.segmentation_only = False
        model = NERModel(base_args.ner, feature_extractor=True)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        word_ids = encoded_essay['word_id_tensor']
        x = torch.stack((encoded_text, attention_mask, word_ids), dim=1).to(model.device)
        output = model.inference(x)
        assert(output.size() == (1, base_args.ner.essay_max_tokens, 9))
