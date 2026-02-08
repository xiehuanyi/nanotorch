from nanotorch import Tensor
from nanotorch.nn import Module, Linear, ReLU
from nanotorch.loss import CrossEntropyLoss
from nanotorch.optimizer import SGD
from nanotorch.dataset import Dataset, DataLoader
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class MNISTModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(784, 128)
        self.relu = ReLU()
        self.l2 = Linear(128, 10)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

def load_data():
    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def train():
    X_train, X_test, y_train, y_test = load_data()
    ds_tr = Dataset(X_train, y_train)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    ds_test = Dataset(X_test, y_test)
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=False)
    
    model = MNISTModel()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1) # Higher LR for single batch
    
    for epoch in range(50):
        for x, y in dl_tr:
            # print(x.shape, y.shape)
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
        # Accuracy
        preds = np.argmax(logits.data, axis=1)
        if isinstance(y, Tensor):
            y = y.data
        correct = (preds == y).sum()
        acc = correct / len(y)
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}, Accuracy: {acc:.4f}")
    
    correct, total = 0, 0
    for x, y in dl_test:
        logits = model(x)
        preds = np.argmax(logits.data, axis=1)
        if isinstance(y, Tensor):
            y = y.data
        correct += (preds == y).sum()
        total += len(y)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()
