from torch.utils.data import TensorDataset

from core.bc import *

# def test_make_bc_dataset(dataset_with_ner_probs, seq_env):
#     dataset = make_bc_dataset(dataset_with_ner_probs, seq_env)
#     sample = dataset[0]
#     assert(isinstance(dataset, TensorDataset))
#     assert(sample[0].size() == (1024,))
#     assert(sample[1].size() == (32, 8))
#     assert(sample[2].size() == (1024, 9))
#     assert(sample[3].size() == ())


