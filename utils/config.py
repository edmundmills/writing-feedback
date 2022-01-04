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

def get_config(filename, args):
    with initialize(config_path=f'../{filename}'):
        cfg = compose('config.yaml', overrides=args.overrides)
    print(OmegaConf.to_yaml(cfg))        
    return cfg

def flatten_args(args):
    return flatten(OmegaConf.to_container(args, resolve=True), reducer='dot')

class WandBRun:
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        project = self.args.name
        if self.args.debug:
            project += '-debug'
        if self.args.wandb:
            wandb.init(
                entity='writing-feedback',
                project=project,
                notes="",
                config=flatten_args(self.args),
            )

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.args.wandb:
            wandb.finish()