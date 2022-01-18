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

def get_config(filename, args=None):
    if isinstance(args, list):
        overrides = args
    elif args is None:
        overrides = []
    else:
        overrides = args.overrides
    path = '../conf/'
    with initialize(config_path=path):
        cfg = compose(f'{filename}.yaml', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))        
    return cfg

def flatten_args(args):
    return flatten(OmegaConf.to_container(args, resolve=True), reducer='dot')

class WandBRun:
    def __init__(self, args, project_name):
        self.args = args
        self.project_name = project_name

    def __enter__(self):
        if self.args.debug:
            self.project_name += '-debug'
        tensorboard = OmegaConf.select(self.args, "sync_tensorboard",
                                       default=False)
        if self.args.wandb:
            wandb.init(
                entity='writing-feedback',
                project=self.project_name,
                notes="",
                sync_tensorboard=tensorboard,
                config=flatten_args(self.args),
            )

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.args.wandb:
            wandb.finish()