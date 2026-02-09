# NanoTorch 🔥

一个用纯NumPy实现的轻量级深度学习框架，模仿PyTorch的API设计，让你深入理解深度学习的底层原理。

## ✨ 特性

- **🔢 Tensor类**: 支持自动求导的张量数据结构
- **🧠 神经网络模块**: Linear、ReLU、BatchNorm2D等常用层
- **⚡ 优化器**: SGD优化器，支持参数更新
- **📉 损失函数**: CrossEntropyLoss、MSELoss等损失函数
- **📊 数据处理**: Dataset和DataLoader，方便批量训练
- **🛠️ 工具函数**: arange等张量操作函数

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/huanyi/nanotorch.git
cd nanotorch
pip install -e .
```

### 基本使用

```python
import nanotorch as nt
from nanotorch.nn import Module, Linear, ReLU
from nanotorch.optimizer import SGD
from nanotorch.loss import CrossEntropyLoss

# 定义模型
class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet()
optimizer = SGD(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

# 前向传播
x = nt.randn(32, 784)  # batch_size=32, input_dim=784
y = nt.randint(0, 10, (32,))  # 10个类别

# 训练步骤
logits = model(x)
loss = criterion(logits, y)

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.data:.4f}")
```

## 📁 项目结构

```
nanotorch/
├── nanotorch/          # 核心库代码
│   ├── __init__.py     # 包初始化
│   ├── tensor.py       # Tensor类，支持自动求导
│   ├── nn.py          # 神经网络模块
│   ├── optimizer.py    # 优化器
│   ├── loss.py        # 损失函数
│   ├── dataset.py     # 数据处理
│   └── utils.py       # 工具函数
├── examples/           # 示例代码
│   └── train_mnist.py # MNIST训练示例
├── test/              # 单元测试
├── pyproject.toml     # 项目配置
└── README.md         # 项目文档
```

## 🎯 核心组件

### Tensor

支持自动求导的张量类：

```python
import nanotorch as nt

# 创建张量
x = nt.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = nt.tensor([4.0, 5.0, 6.0], requires_grad=True)

# 数学运算
z = x * y + 2
z.backward()  # 自动求导

print(x.grad)  # [4.0, 5.0, 6.0]
print(y.grad)  # [1.0, 2.0, 3.0]
```

### 神经网络模块

提供常用的神经网络层：

```python
from nanotorch.nn import Module, Linear, ReLU, BatchNorm2D

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.bn1 = BatchNorm1D(256)
        self.relu1 = ReLU()
        self.fc2 = Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
```

### 优化器

SGD优化器实现：

```python
from nanotorch.optimizer import SGD

model = MyModel()
optimizer = SGD(model.parameters(), lr=0.01)

# 训练循环
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
```

## 📈 示例：MNIST手写数字识别

运行MNIST训练示例：

```bash
cd examples
python train_mnist.py
```

**性能表现：**
- 测试准确率：**97.60%**

## 🔬 消融实验

我们进行了模型架构和优化器的消融实验，结果如下：

### 实验配置
- **数据集**: MNIST (784特征, 10类别)
- **训练轮数**: 5
- **批次大小**: 64
- **测试集比例**: 10%
- **随机种子**: 42

### 模型架构对比

| 模型架构 | SGD | Adam |
|---------|-----|------|
| Linear | 91.69% | 92.00% |
| Linear+ReLU | 96.50% | 97.50% |
| Linear+BN | 91.13% | 91.66% |
| Linear+ReLU+BN | 97.07% | 97.20% |

### 关键发现

1. **ReLU激活函数显著提升性能** (+5%准确率提升)
2. **BatchNorm单独使用效果不佳**，但与ReLU组合效果最佳
3. **Adam优化器在大多数情况下略优于SGD**
4. **最佳组合**: Linear+ReLU+Adam (97.50%准确率)
5. **最差组合**: Linear+BN+SGD (91.13%准确率)

### 性能排名

1. Linear+ReLU+Adam: **97.50%**
2. Linear+ReLU+BN+Adam: **97.20%**
3. Linear+ReLU+BN+SGD: **97.07%**
4. Linear+ReLU+SGD: **96.50%**
5. Linear+Adam: **92.00%**
6. Linear+SGD: **91.69%**
7. Linear+BN+Adam: **91.66%**
8. Linear+BN+SGD: **91.13%**

### 运行消融实验

```bash
# 运行所有消融实验
cd examples
python train_mnist_ablation.py --ablation

# 运行特定实验
python train_mnist_ablation.py --model linear_relu --optimizer adam

# 查看所有可用选项
python train_mnist_ablation.py --help
```

详细实验日志请查看: `logs/mnist_ablation_results.log`

## 🧪 运行测试

```bash
# 运行所有测试
python test/run_tests.py

# 运行特定测试
python test/test_tensor.py
python test/test_nn.py
```

## 🎓 学习价值

NanoTorch是一个教育性项目，旨在帮助理解深度学习框架的核心原理：

1. **自动求导机制**：理解梯度如何在计算图中反向传播
2. **张量操作**：学习基础数学运算的实现
3. **神经网络层**：掌握各种层的forward和backward实现
4. **优化器原理**：了解参数更新算法
5. **模块化设计**：学习如何构建可扩展的框架

## 📊 性能对比

| 特性 | NanoTorch | PyTorch |
|------|-----------|---------|
| 自动求导 | ✅ | ✅ |
| 神经网络层 | ✅ | ✅ |
| 优化器 | ✅ | ✅ |
| GPU支持 | ❌ | ✅ |
| 性能 | 基础 | 高性能 |
| 学习难度 | 简单 | 中等 |

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

⭐ 如果这个项目对你有帮助，请给个Star！
