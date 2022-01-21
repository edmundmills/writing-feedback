from pathlib import Path

data_path = Path('data')
essay_dir = data_path / 'train'
label_file = data_path / 'train.csv'
ner_probs_path = data_path / 'ner_probs.pkl'

de_type_to_num = {
            'None': 0,
            'Lead': 1,
            'Position': 2,
            'Claim': 3,
            'Counterclaim': 4,
            'Rebuttal': 5,
            'Evidence': 6,
            'Concluding Statement': 7
        }

de_num_to_type = [k for k, _ in sorted(de_type_to_num.items(), key=lambda x : x[1])]

