from core.models.essay_feedback import *

def test_positional_encoding():
    pos_encoder = PositionalEncoder(32)
    input_tokens = torch.rand(20, 768)
    pos_encoded = pos_encoder(input_tokens)
    assert(pos_encoded.size() == input_tokens.size())

class TestEssayModel:
    def test_encode(self, essay_feedback_args, d_elem_encoder, essay):
        essay_model = EssayModel(essay_feedback_args, d_elem_encoder=d_elem_encoder)
        d_elems = essay.d_elems_text
        encoded_essay = essay_model.encode(d_elems)
        assert(encoded_essay.size() == (essay_feedback_args.max_discourse_elements, 768))

    def test_inference(self, essay_model, essay):
        preds = essay_model.inference(essay.text, essay.pstrings)
        assert(preds.size() == (1, 32, 8))
        assert(torch.sum(preds[:,0,:]).item() == 1)
        assert(round(torch.sum(preds).item()) == 32)