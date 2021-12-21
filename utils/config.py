import argparse

from hydra import compose, initialize
from flatten_dict import flatten
from omegaconf import OmegaConf
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()
    return args

def get_config(args):
    with initialize(config_path='../config'):
        cfg = compose('config.yaml', overrides=args.overrides)
    print(OmegaConf.to_yaml(cfg))        
    return cfg

def flatten_args(args):
    return flatten(OmegaConf.to_container(args, resolve=True), reducer='dot')

class wandb_run:
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        if not self.args.debug:
            wandb.init(
                entity='writing-feedback',
                project=self.args.name,
                notes="",
                config=flatten_args(self.args),
            )

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self.args.debug:
            wandb.finish()