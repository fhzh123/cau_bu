import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer: str, input_ids_list: list, label_list: list,
                 attention_mask_list: list, token_type_ids_list: list = None, 
                 min_len: int = 4, max_len: int = 512):
        data = []
        if tokenizer in ['T5', 'Bart']:
            for i, a, l in zip(input_ids_list, attention_mask_list, label_list):
                if min_len <= len(i) <= max_len:
                    data.append((i, a, l))
        else:
            for i, t, a, l in zip(input_ids_list, token_type_ids_list, 
                                  attention_mask_list, label_list):
                if min_len <= len(i) <= max_len:
                    data.append((i, t, a, l))
        
        self.tokenizer = tokenizer
        self.data = tuple(data)
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        if self.tokenizer == 'T5':
            id_, attention_, label_ = self.data[index]
            return id_, attention_, label_
        else:
            id_, token_type_, attention_, label_ = self.data[index]
            return id_, token_type_, attention_, label_
    
    def __len__(self):
        return self.num_data