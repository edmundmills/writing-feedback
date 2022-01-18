from pathlib import Path

data_path = Path('data')
essay_dir = data_path / 'train'
label_file = data_path / 'train.csv'

argument_types = {
            'None': 0,
            'Lead': 1,
            'Position': 2,
            'Claim': 3,
            'Counterclaim': 4,
            'Rebuttal': 5,
            'Evidence': 6,
            'Concluding Statement': 7
        }

argument_names = [k for k, _ in sorted(argument_types.items(), key=lambda x : x[1])]

ner_token_names = [
    'None',
    'Lead (start)',
    'Position (start)',
    'Claim (start)',
    'Counterclaim (start)',
    'Rebuttal (start)',
    'Evidence (start)',
    'Concluding Statement (start)',
    'Lead (cont)',
    'Position (cont)',
    'Claim (cont)',
    'Counterclaim (cont)',
    'Rebuttal (cont)',
    'Evidence (cont)',
    'Concluding Statement (cont)',
]