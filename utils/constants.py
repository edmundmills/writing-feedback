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

ner_num_to_token = [
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
    'Concluding Statement (cont)'
]

ner_token_to_num = {name: idx for idx, name
                    in enumerate(ner_num_to_token)}