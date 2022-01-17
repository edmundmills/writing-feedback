from core.classification import *


class TestClassificationModel:
    def test_encode(self, kls_args, d_elem_encoder, essay):
        essay_model = ClassificationModel(kls_args, d_elem_encoder=d_elem_encoder)
        d_elems = essay.d_elems_text
        encoded_essay = essay_model.encode(d_elems)
        assert(encoded_essay.size() == (kls_args.max_discourse_elements, 769))

#     def test_inference(self, kls_model, essay):
#         preds = kls_model.inference(essay.text, essay.pstrings)
#         assert(preds.size() == (1, 32, 8))
#         assert(round(torch.sum(preds[:,0,:]).item()) == 1)
#         assert(round(torch.sum(preds).item()) == 32)