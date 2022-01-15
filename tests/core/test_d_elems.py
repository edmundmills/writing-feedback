from core.d_elems import *
from tests.conftest import kls_args


class TestTokenizer:
    def test_valid(self, kls_args, essay):
        tokenizer = DElemTokenizer(kls_args)
        tokenized = tokenizer.tokenize(essay.d_elems_text)
        print(tokenized)
        assert(tokenized['input_ids'].shape == (32, 768))


class TestDElemEncoder:
    def test_valid(self, d_elem_tokenizer, kls_args, essay):
        encoder = DElemEncoder(kls_args)
        tokenized = d_elem_tokenizer.tokenize(essay.d_elems_text)
        output = encoder.encode(tokenized)
        assert(output.size() == (32, 769))