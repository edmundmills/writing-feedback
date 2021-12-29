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
