from core.ner import *


class TestEncode:
    def test_valid(self, ner_args, essay):
        tokenizer = NERTokenizer(ner_args)
        tokenized = tokenizer.encode(essay.text)
        assert(tokenized['input_ids'].size() == (1, 1024))
        assert(tokenized['attention_mask'].size() == (1, 1024))
        assert(len(tokenized['word_ids']) == 1024)
        assert(tokenized['word_id_tensor'].size() == (1,1024))


class TestSegmentationModel:
    def test_forward_ner(self, encoded_essay, ner_args):
        model = NERModel(ner_args)
        encoded_text = encoded_essay['input_ids']
        attention_mask = encoded_essay['attention_mask']
        output = model(encoded_text.to(model.device),
                       attention_mask.to(model.device))
        assert(output.size() == (1, ner_args.essay_max_tokens, 2))
