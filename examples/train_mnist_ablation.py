from nanotorch import Tensor
from nanotorch.nn import Module, Linear, ReLU, BatchNorm1d
from nanotorch.loss import CrossEntropyLoss
from nanotorch.optimizer import SGD, Adam
from nanotorch.dataset import Dataset, DataLoader
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import argparse

class LinearModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(784, 10)
    
    def forward(self, x):
        x = self.l1(x)
        return x

class LinearReLUModel(Module):
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

class LinearBNModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(784, 128)
        self.bn = BatchNorm1d(128)
        self.l2 = Linear(128, 10)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = self.l2(x)
        return x

class LinearReLUBNModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(784, 128)
        self.relu = ReLU()
        self.bn = BatchNorm1d(128)
        self.l2 = Linear(128, 10)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.bn(x)
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

def train(model_name, optimizer_name, epochs=10):
    X_train, X_test, y_train, y_test = load_data()
    ds_tr = Dataset(X_train, y_train)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    ds_test = Dataset(X_test, y_test)
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=False)
    
    # Model selection
    if model_name == "linear":
        model = LinearModel()
    elif model_name == "linear_relu":
        model = LinearReLUModel()
    elif model_name == "linear_bn":
        model = LinearBNModel()
    elif model_name == "linear_relu_bn":
        model = LinearReLUBNModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Optimizer selection
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    criterion = CrossEntropyLoss()
    
    print(f"Training {model_name} with {optimizer_name}")
    for epoch in range(epochs):
        for x, y in dl_tr:
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
    
    # Test accuracy
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
    return acc

def run_ablation_experiments():
    models = ["linear", "linear_relu", "linear_bn", "linear_relu_bn"]
    optimizers = ["sgd", "adam"]
    
    results = {}
    
    for model_name in models:
        for optimizer_name in optimizers:
            key = f"{model_name}_{optimizer_name}"
            print(f"\n{'='*50}")
            print(f"Running experiment: {key}")
            print(f"{'='*50}")
            
            try:
                test_acc = train(model_name, optimizer_name, epochs=5)  # Reduced epochs for faster experiments
                results[key] = test_acc
                print(f"Final test accuracy for {key}: {test_acc:.4f}")
            except Exception as e:
                print(f"Error in {key}: {e}")
                results[key] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for model_name in models:
        print(f"\n{model_name.upper()}:")
        for optimizer_name in optimizers:
            key = f"{model_name}_{optimizer_name}"
            acc = results[key]
            if acc is not None:
                print(f"  {optimizer_name:8s}: {acc:.4f}")
            else:
                print(f"  {optimizer_name:8s}: FAILED")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Training with Ablation Studies')
    parser.add_argument('--model', type=str, choices=['linear', 'linear_relu', 'linear_bn', 'linear_relu_bn'], 
                        help='Model architecture')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], 
                        help='Optimizer type')
    parser.add_argument('--ablation', action='store_true', 
                        help='Run all ablation experiments')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_experiments()
    elif args.model and args.optimizer:
        train(args.model, args.optimizer, args.epochs)
    else:
        print("Please specify --model and --optimizer, or use --ablation to run all experiments")
        print("Available models: linear, linear_relu, linear_bn, linear_relu_bn")
        print("Available optimizers: sgd, adam")