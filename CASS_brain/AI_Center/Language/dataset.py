import os
import json
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        data_path = os.path.join(path, 'chat.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[str(idx)]
        x_input_ids = self.tokenizer(text[0], 
                                   return_tensors='pt', 
                                   truncation=True, 
                                   padding='max_length',
                                   max_length=self.max_len)
        y_input_ids, _, _ = self.tokenizer(text[1], 
                                   return_tensors='pt', 
                                   truncation=True, 
                                   padding='max_length',
                                   max_length=self.max_len).values()
        labels = y_input_ids.clone()
        labels[:, :-1] = y_input_ids[:, 1:]  # Right shift
        labels[:, -1] = -100 
        return x_input_ids, labels
    