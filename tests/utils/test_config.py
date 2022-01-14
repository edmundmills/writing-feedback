from utils.config import *

def test_segmentation():
    args = get_config('segmentation') 
    assert(args.ner.name == 'ner')