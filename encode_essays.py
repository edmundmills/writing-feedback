from core.models.argument_encoder import ArgumentModel
from core.dataset import EssayDataset

if __name__ == '__main__':
    model_name = ''
    
    essay_dataset = EssayDataset()

    arg_model = ArgumentModel().load(model_name)

    essay = essay_dataset[0]
    encoded_essay = arg_model.encode_essay(essay)
    print(encoded_essay)