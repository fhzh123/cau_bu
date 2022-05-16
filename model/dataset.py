import random
# Import PyTorch
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
        self.data = list(data)
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

class PadCollate:
    def __init__(self, tokenizer, pad_index=0, sep_index=3, dim=0):
        self.dim = dim
        self.pad_index = pad_index
        self.sep_index = sep_index
        self.tokenizer = tokenizer

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences
        
        out_ = zip(*batch)
        if self.tokenizer in ['T5', 'Bart']:
            id_, attention_, label_ = out_
            return pack_sentence(id_), \
                pack_sentence(attention_), torch.LongTensor(label_)
        else:
            id_, token_type_, attention_, label_ = out_
            return pack_sentence(id_), pack_sentence(token_type_), \
                pack_sentence(attention_), torch.LongTensor(label_)
        
    def __call__(self, batch):
        return self.pad_collate(batch)