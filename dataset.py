import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BartTokenizer

class WeakSupDataset(Dataset):
    def __init__(self, split, n_docs=None):
        data_path = f'/HDD/dataset/CNN_WS/{split}.json'

        docs = json.load(open(data_path))
        docs = docs[:n_docs]

        self._examples = []
        for doc in docs:
            for asp_sum in doc['aspect_summaries']:
                self._examples.append({
                    'aspect': asp_sum['aspect'],
                    'wiki_words': asp_sum['wiki_words'],
                    'document': doc['document'],
                    'summary': asp_sum['summary']
                })

    def __getitem__(self, item):
        example = self._examples[item]
        return {
            'src': '{aspect} : {wiki_words}\n\n{doc}'.format(
                aspect=example['aspect'],
                wiki_words=' '.join(example['wiki_words']),
                doc=example['document']),
            'tgt': example['summary']
        }

    def __len__(self):
        return len(self._examples)

class MANewsDataset(Dataset):
    def __init__(self, split, supervisor, n_wiki_words, n_docs=None):
        data_path = f'data/manews/{split}.json'
        if not os.path.exists(data_path):
            os.system('bash data_utils/download_manews.sh')

        self._examples = json.load(open(data_path))[:n_docs]
        for example in tqdm(self._examples, desc=f'loading manews {split} set'):
            example['wiki_words'] = supervisor.get_wiki_words(
                aspect=example['aspect'], document=example['document'],
                n_limit=n_wiki_words) if supervisor is not None else []

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        example = self._examples[item]
        return {
            'src': '{aspect} : {wiki_words}\n\n{doc}'.format(
                aspect=example['aspect'],
                wiki_words=' '.join(example['wiki_words']),
                doc=example['document']),
            'tgt': example['summary']
        }
