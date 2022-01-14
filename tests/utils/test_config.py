from omegaconf import OmegaConf 

from utils.config import *

def test_segmentation():
    args = get_config('segmentation') 
    assert(args.ner.name == 'ner')
    args.sync_tensorboard = True
    assert(OmegaConf.select(args, 'sync_tensorboard'))