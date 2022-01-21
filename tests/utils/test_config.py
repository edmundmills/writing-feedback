from omegaconf import OmegaConf 

from utils.config import *

def test_segmentation():
    args = get_config('base') 
    assert(args.ner.name != '')
    args.rl.sync_tensorboard = True
    assert(OmegaConf.select(args.rl, 'sync_tensorboard'))