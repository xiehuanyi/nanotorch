import numpy as np
from nanotorch.tensor import Tensor

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_x = [self.dataset[j][0] for j in batch_indices]
            batch_y = [self.dataset[j][1] for j in batch_indices]
            yield Tensor(np.array(batch_x)), Tensor(np.array(batch_y), dtype=np.int64)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size