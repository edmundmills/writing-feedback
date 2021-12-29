from typing import List

from core.model import Model

class Segmenter(Model):
    def __init__(self):
        super().__init__()
        # self.model = Longformer

    def forward(self, text:str) -> List[str]:
        pass

    def train(self, test_dataset, val_dataset) -> None:
        """
        Supervised NER or RL NER
        """