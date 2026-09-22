# 🚀 动手学深度学习 (D2L) 实战与手搓神经网络

本项目记录我学习深度学习与自然语言处理（NLP）的代码、手搓实现与实验 Notebook。

## 📖 关于本仓库的说明

**内容分为两类，已在下方索引中标注：**

| 标注 | 含义 |
|:--:|---|
| 📘 **教材代码 + 我的笔记** | 代码主体来自《动手学深度学习》(D2L) 教材，我在练习过程中**添加了中文理解笔记与配图注解**——记录推导过程、直觉理解、易错点 |
| ✍️ **自己的实验** | 我自己想做的实验与实现，代码独立编写 |

**我的笔记长什么样**：在教材代码的对应位置，我会写下自己的理解。例如：

- 权重衰退一节：「防止过拟合思路：1. 限制模型复杂度 —— 1.1 控制参数数量 1.2 控制参数大小（即权重衰退）……」
- RNN 一节：「结合总体梯度公式理解：梯度向量 G = [0.3, 8.7, 0.5, -0.2]……」
- MLP 一节：「感知机的不足 → 解决 XOR 问题：使用两条不同的线性分类器……」

**我的学习目标**：先把每个模块的数学推导和实现细节搞明白，再脱离教材独立实现。目前进度到 Attention / Transformer（详见第三节）。

## 📑 学习大纲与 Notebook 索引

### 🔹 一、 基础与经典模型 📘
- [`d2l_2data_processing.ipynb`](./d2l_2data_processing.ipynb) - 数据预处理与张量操作
- [`d2l_3linear algebra.ipynb`](./d2l_3linear%20algebra.ipynb) - 线性代数基础
- [`d2l_4matrix and atuo gradient.ipynb`](./d2l_4matrix%20and%20atuo%20gradient.ipynb) - 矩阵与自动求导
- [`d2l_5linear_regression.ipynb`](./d2l_5linear_regression.ipynb) - 线性回归
- [`d2l_6softmax回归.ipynb`](./d2l_6softmax回归.ipynb) - Softmax 分类
- [`d2l_7mlp.ipynb`](./d2l_7mlp.ipynb) - 多层感知机 (MLP)｜笔记：感知机局限、XOR 问题的线性分割解法

### 🔹 二、 循环神经网络与序列模型 (RNN & Seq2Seq) 📘
- [`d2l_8 序列模型.ipynb`](./d2l_8%20序列模型.ipynb) - 序列模型与文本预处理
- [`d2l_9 RNN.ipynb`](./d2l_9%20RNN.ipynb) - 循环神经网络基础｜笔记：潜变量模型、困惑度指标、梯度向量分析
- [`d2l_10 GRU.ipynb`](./d2l_10%20GRU.ipynb) - 门控循环单元 (GRU)
- [`d2l_11 LSTM.ipynb`](./d2l_11%20LSTM.ipynb) - 长短期记忆网络 (LSTM)
- [`seq2seq.ipynb`](./seq2seq.ipynb) - 编码器-解码器与机器翻译基线｜笔记：评价指标 BLEU
- [`束搜索.ipynb`](./束搜索.ipynb) - Beam Search 解码生成策略｜笔记：局部最优与全局最优、时间复杂度分析

### 🔹 三、 注意力机制与 Transformer (Attention & Transformer) 📘
- [`attention.ipynb`](./attention.ipynb) - 注意力机制基础 (Nadaraya-Watson / 评分函数)
- [`自注意力和位置编码.ipynb`](./自注意力和位置编码.ipynb) - Self-Attention 与 Positional Encoding
- [`transformer.ipynb`](./transformer.ipynb) - 完整 Transformer 架构组装与训练
- [`transformer_modules.py`](./transformer_modules.py) - Transformer 模块化整理（含 masked softmax、缩放点积注意力、多头注意力、位置编码）

### 🔹 四、 自己的实验（✍️ 独立实现）

> 以下 Notebook 不是教材章节练习，而是我在学习过程中自己想验证的问题。
> 代码独立编写，用于把"看得懂"变成"写得出"。

| Notebook | 我在验证什么 |
|---|---|
| [`手搓一个单层神经网络并训练.ipynb`](./手搓一个单层神经网络并训练.ipynb) | 不用 `nn.Linear`，用张量运算手写一层网络的前向与反向，跑通训练 |
| [`从单层网络到深层网络.ipynb`](./从单层网络到深层网络.ipynb) | 逐层堆叠加深网络，观察深层网络的信息传递与梯度变化 |
| [`手搓一个参数衰退.ipynb`](./手搓一个参数衰退.ipynb) | 手动实现 L2 权重衰退与 L1 稀疏化，对比不同 λ 对过拟合的影响 |
| [`张量实验&CNN尺寸推导.ipynb`](./张量实验&CNN尺寸推导.ipynb) | 用代码验证卷积输出尺寸公式与张量维度变换 |
| [`bsaic practice.ipynb`](./bsaic%20practice.ipynb) | 基础语法与张量操作的练习 |

### 🔹 五、 我的理解笔记示例

学习过程中形成的一些笔记（写在各 Notebook 的 Markdown cell 中）：

- **权重衰退**：「限制模型复杂度有两条路——控制参数数量（模型大小）、控制参数大小（权重衰退）。后者通过限制参数取值范围来控制模型容量。」
- **RNN 梯度**：「结合总体梯度公式理解：梯度向量 G = [0.3, 8.7, 0.5, -0.2]，可见某一维的梯度远大于其他维度时，参数更新会被这一维主导。」
- **MLP 与 XOR**：「感知机的不足在于只能做线性分割；解决 XOR 需要用两条不同的线性分类器组合。」
- **Beam Search**：「在前后状态有依赖的前提下，局部最优 ≠ 全局最优。每次只选一个分支的贪心策略时间复杂度为 O(t)，束搜索用宽度 k 换取更好的全局解。」

---

## 🎯 学习进度与下一步

**已完成**：数据预处理 / 线性代数 / 自动求导 / 线性回归 / Softmax / MLP / 序列模型 / RNN / GRU / LSTM / Seq2Seq / Beam Search / Attention / Self-Attention / 位置编码 / Transformer

**当前正在做**：把 Transformer 拆成可复用的模块（见 `transformer_modules.py`），理解每个子模块的输入输出维度与 mask 机制。

**下一步计划**：脱离教材，用 PyTorch 独立从零实现训练循环、CNN 与 Transformer Encoder，并迁移到一个实际项目上。