# Karpathy MicroGPT 源码解读报告

> 基于 gpt.py（243行），纯 Python 实现的 GPT 模型，零依赖。
> 本报告结合逐行学习过程中的提问与讨论整理而成。

---

## 一、整体架构

```
gpt.py 的结构：

L1-14    数据加载与随机种子
L25-31   分词器（Tokenizer）
L35-115  Value 类（自动求导引擎）
L117-133 模型参数初始化
L137-149 工具函数（linear / softmax / rmsnorm）
L151-187 gpt() 前向传播函数
L189-227 训练循环（正向 → 算loss → 反向 → Adam更新）
L229-243 推理（逐字母生成名字）
```

完整的数据流：

```
训练: 名字 → 分词 → gpt()前向 → 算loss → backward()反向 → Adam更新权重 → 重复500次
推理: BOS → gpt()前向 → softmax → 随机采样 → 输出字母 → 重复直到BOS
```

---

## 二、数据加载与分词器（L1-31）

### L9-14：导入与随机种子

```python
import os       # 文件操作
import math     # log, exp 数学函数
import random   # 随机数

random.seed(42) # 固定随机种子，保证每次运行结果一致
```

### L17-23：加载数据

```python
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
```

从 input.txt 读取约 32,000 个英文名字（如 "emma", "olivia", "jan"），打乱顺序。

### L26-30：分词器

```python
chars = ['<BOS>'] + sorted(set(''.join(docs)))  # 27个token: BOS + a~z
stoi = { ch:i for i, ch in enumerate(chars) }   # 字符→数字: {'<BOS>':0, 'a':1, ..., 'z':26}
itos = { i:ch for i, ch in enumerate(chars) }   # 数字→字符: {0:'<BOS>', 1:'a', ..., 26:'z'}
BOS = stoi['<BOS>']                              # BOS = 0
```

字符级分词器，词汇表大小 = 27（BOS + 26个字母）。

---

## 三、Value 类——自动求导引擎（L35-115）

### 核心思想

Value 类把每个数字包装成一个节点，记录它是怎么算出来的（计算图）。这样就能从最终的 loss 出发，沿着计算图反向传播，自动算出每个参数的梯度。

### `__init__`（L38-43）

```python
def __init__(self, data, _children=(), _op=''):
    self.data = data           # 实际的数值
    self.grad = 0              # 梯度，初始为0
    self._backward = lambda: None  # 反向传播函数，默认什么都不做
    self._prev = set(_children)    # 父节点（产生这个值的输入节点）
    self._op = _op             # 运算符，用于调试
```

**Q: 什么是梯度（grad）？**
梯度表示"这个参数增大一点点，最终的 loss 会怎么变"。grad=2 意味着参数增大1，loss增大2。训练时根据梯度方向调整参数，让 loss 变小。

**Q: 为什么 `_backward` 前面有下划线？**
Python 惯例，下划线开头表示"内部使用，外部不要直接调用"。用户应该调用 `backward()`（无下划线），它会自动按正确顺序调用所有节点的 `_backward`。

**Q: `lambda: None` 是什么意思？**
一个什么都不做的函数。叶子节点（如初始参数）没有反向传播逻辑，所以用空函数占位。

**Q: `_children` 为什么叫 children 而不叫 parent？**
命名确实容易混淆。从计算图的角度看，`c = a + b` 中 a 和 b 是 c 的"子节点"（children），因为它们是构成 c 的输入。但从数据流的角度看，a 和 b 更像"父节点"。这里用的是计算图的命名惯例。

### `__add__`（L45-52）——加法与反向传播

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad += out.grad    # 加法的梯度：直接传递
        other.grad += out.grad   # 两个输入都得到相同的梯度
    out._backward = _backward
    return out
```

加法的梯度规则：`c = a + b`，a 增大1 → c 增大1，所以 `a.grad += c.grad`。b 同理。用 `+=` 而不是 `=`，是因为同一个变量可能被用多次（如 `c = a + a`），梯度需要累加。

**Q: 为什么 `grad=1.0` 在示例中？实际训练中 grad 怎么确定？**
`backward()` 方法（L92-106）会自动设置 `self.grad = 1`（L104），然后沿计算图反向传播。1.0 是起点——"loss 对自己的梯度是1"。

### `__mul__`（L54-61）——乘法

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad += other.data * out.grad   # a的梯度 = b的值 × 输出梯度
        other.grad += self.data * out.grad   # b的梯度 = a的值 × 输出梯度
    out._backward = _backward
    return out
```

乘法的梯度规则：`c = a × b`，a.grad = b 的值，b.grad = a 的值。这就是为什么 b 很大时，a 的梯度也很大——b=100 意味着 a 的微小变化会被放大100倍。

### `__pow__`（L63-69）——幂运算

梯度规则：`c = a^n`，`a.grad = n × a^(n-1)`。这是微积分的幂函数求导公式。

### log / exp / relu（L71-90）

| 函数 | 正向 | 反向梯度 |
|------|------|----------|
| `log(x)` | `math.log(x)` | `1/x` |
| `exp(x)` | `math.exp(x)` | `exp(x)` 本身 |
| `relu(x)` | `max(0, x)` | x>0 时为1，x≤0 时为0 |

### `backward()`（L92-106）——反向传播入口

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):          # 拓扑排序：保证先算"后面的"再算"前面的"
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1               # 起点：loss对自己的梯度=1
    for v in reversed(topo):    # 反向遍历，依次调用每个节点的_backward
        v._backward()
```

### 运算符快捷方式（L108-115）

```python
def __neg__(self): return self * -1          # -a
def __sub__(self, other): return self + (-other)  # a - b
def __truediv__(self, other): return self * other**-1  # a / b
# ... 等等
```

所有运算都转换成 `__add__`、`__mul__`、`__pow__` 的组合，不需要单独实现反向传播。

---

## 四、模型参数初始化（L117-133）

### 超参数

```python
n_embd = 16      # 嵌入维度：每个字母用16个数表示
n_head = 4       # 注意力头数：4个头，每个头看4维
n_layer = 1      # 层数：只有1层（大模型有几十层）
block_size = 8   # 最大序列长度：一次最多处理8个字母
head_dim = 4     # 每个头的维度：16 / 4 = 4
```

### 参数矩阵

```python
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

用高斯随机数初始化矩阵，标准差0.02（很小的随机数）。

所有权重矩阵一览：

| 矩阵名 | 大小 | 用途 |
|--------|------|------|
| `wte` | 27×16 | 字母嵌入表：字母编号 → 16维向量 |
| `wpe` | 8×16 | 位置嵌入表：位置编号 → 16维向量 |
| `attn_wq` | 16×16 | 生成 Q（用于算相似度） |
| `attn_wk` | 16×16 | 生成 K（用于被匹配） |
| `attn_wv` | 16×16 | 生成 V（用于携带信息） |
| `attn_wo` | 16×16 | 融合多头注意力的输出 |
| `mlp_fc1` | 64×16 | MLP 升维：16→64 |
| `mlp_fc2` | 16×64 | MLP 降维：64→16 |
| `lm_head` | 27×16 | 输出层：16维→27个字母分数 |

总参数量：4064 个浮点数。

**Q: wpe 的维度为什么是 8×16？**
`block_size=8`，最多处理8个位置，所以位置嵌入表有8行。每行16维，和字母嵌入维度一致，这样才能相加。

**Q: 只有 wte 和 wpe 就够了吗？**
嵌入表只有这两个。但模型还有注意力和MLP的权重矩阵，它们负责后续的信息加工。嵌入只是第一步——把字母变成向量。

---

## 五、工具函数（L137-149）

### linear（L137-138）——矩阵乘法（线性变换）

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

本质就是矩阵乘法：x 和 w 的每一行做点积。

**点积**是唯一需要记住的运算：对应位置相乘，再全部加起来。
```
a = [1, 2, 3]
b = [4, 5, 6]
点积 = 1×4 + 2×5 + 3×6 = 32
```

矩阵乘法 = 做很多次点积。矩阵有几行，输出就有几维：

| 矩阵大小 | 效果 | 代码中的例子 |
|----------|------|-------------|
| 小×大（如16×64） | 降维（压缩） | mlp_fc2: 64→16 |
| 大×小（如64×16） | 升维（展开） | mlp_fc1: 16→64 |
| 同×同（如16×16） | 同维变换（旋转） | attn_wq: 16→16 |

linear 在 gpt() 中有四种用法：
1. **同维变换**（16→16）：生成 Q/K/V，维度不变但信息变了
2. **升维**（16→64）：MLP 第一层，展开信息以便逐维处理
3. **降维**（64→16）：MLP 第二层，压缩回统一维度
4. **输出映射**（16→27）：lm_head，映射到词汇表大小

### softmax（L140-144）——"软选择"，把分数变成概率

```python
def softmax(logits):
    max_val = max(val.data for val in logits)       # 找最大值
    exps = [(val - max_val).exp() for val in logits] # 减最大值后取e的指数
    total = sum(exps)                                 # 求和
    return [e / total for e in exps]                  # 归一化
```

```
输入: [2.0, 1.0, 0.5]
减最大值: [0.0, -1.0, -1.5]
取指数: [1.0, 0.37, 0.22]
归一化: [0.63, 0.23, 0.14]  ← 全部为正，加起来=1
```

名字由来：soft（软的）+ max（最大值）。普通 max 只选最大的（硬选择），softmax 给每个都分一点概率，大的分多（软选择）。

减去 max_val 是为了数值稳定——防止 `e^x` 在 x 很大时溢出。

**Q: softmax 有替代方案吗？**
- hardmax：最大的给1，其他给0。不可导，训练时不能用
- sigmoid：每个值独立映射到0~1，加起来不等于1
- sparsemax：小的值直接变0，结果更稀疏
- temperature 缩放：推理时在 softmax 前除以温度值，控制分布的集中程度

### rmsnorm（L146-149）——"拉齐"，归一化

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)  # 均方（Mean Square）
    scale = (ms + 1e-5) ** -0.5              # 1/√ms（Root）
    return [xi * scale for xi in x]           # 缩放（Normalize）
```

名字由来：RMS（Root Mean Square，均方根）+ norm（normalize，归一化）。

作用：不管输入的数值是大是小，处理完后都在差不多的范围内。防止数值爆炸或消失。

---

## 六、gpt() 前向传播函数（L151-187）

### 整体流程

```
输入: token_id（字母编号）, pos_id（位置编号）
  ↓
第1步: 嵌入 → 查表得到16维向量
第2步: rmsnorm → 归一化
第3步: 生成Q/K/V → 同一个x乘三个不同矩阵
第4步: 多头切分 → 16维切成4份，每份4维
第5步: 注意力计算 → 点积→softmax→加权求和
第6步: MLP → 升维→激活→降维
第7步: 输出层 → 16维→27个字母分数
```

### 第1步：嵌入（L152-155）

```python
tok_emb = state_dict['wte'][token_id]  # 查字母嵌入表，得到16维
pos_emb = state_dict['wpe'][pos_id]    # 查位置嵌入表，得到16维
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 相加
x = rmsnorm(x)                         # 归一化
```

把"字母编号"这个干巴巴的数字，变成包含身份和位置信息的16维向量。

```
tok_emb = [0.3, -0.1, 0.5, ...]   ← "我是j"
pos_emb = [0.1,  0.2, 0.0, ...]   ← "我在位置1"
x       = [0.4,  0.1, 0.5, ...]   ← "我是位置1的j"
```

### 第2步：残差保存 + 归一化（L159-160）

```python
x_residual = x    # 存副本，后面残差连接用
x = rmsnorm(x)    # 归一化
```

### 第3步：生成 Q、K、V（L161-165）

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])  # 16维→16维
k = linear(x, state_dict[f'layer{li}.attn_wk'])  # 16维→16维
v = linear(x, state_dict[f'layer{li}.attn_wv'])  # 16维→16维
keys[li].append(k)      # 存K到历史列表
values[li].append(v)     # 存V到历史列表
```

同一个 x，乘以三个不同的固定矩阵（训练时学好的），得到三个不同的16维向量。

**Q: Q 只有1个吗？怎么生成的？**
对，处理当前字母时 Q 只有1个。K 和 V 会累积——每处理一个字母就 append 一个。处理第3个字母时：Q 有1个，K 有3个，V 有3个。

**Q: K 和 V 有什么区别？**
K 用来"被匹配"——和 Q 做点积算相似度。V 用来"携带信息"——匹配上之后，真正拿来用的内容。两者从同一个 x 生成，但经过不同的矩阵，所以值不同，各司其职。

**Q: 为什么不存 Q？**
Q 只有当前字母需要用（"我在找什么"），用完就扔。K 和 V 要留着，给后面的字母回头看。

**Q: Q/K/V 和模型权重的关系？**
模型文件里存的是 wq/wk/wv 矩阵（固定的）。Q/K/V 是推理时用矩阵和输入实时算出来的中间结果，用完就扔。类比：矩阵是菜谱（固定），x 是食材（每次不同），Q/K/V 是做出来的菜（临时的）。

### 第4步：多头切分（L166-171）

```python
x_attn = []
for h in range(n_head):                                    # h = 0,1,2,3
    hs = h * head_dim                                       # hs = 0,4,8,12
    q_h = q[hs:hs+head_dim]                                # 取4维
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]          # 每个历史K取对应4维
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]        # 每个历史V取对应4维
```

16维向量切成4份，每份4维。每个头独立处理自己的4维。

```
q = [0.9, 0.4, 0.5, 0.8, | 0.3, 0.7, 0.1, 0.6, | 0.2, 0.5, 0.9, 0.3, | 0.7, 0.1, 0.4, 0.8]
     ----头0(q[0:4])----   ----头1(q[4:8])----     ----头2(q[8:12])---    ----头3(q[12:16])---
```

**Q: 多头切分为什么能做到独立判断？**
wq 矩阵虽然是一个16×16的整体，但不同区域的行在训练时学到了不同的功能。头0用第0~3行的结果，头1用第4~7行的结果，不同的行有不同的参数值，自然关注不同的东西。

**Q: 多头注意力的优势是什么？**
单头只能产生一组 softmax 权重，只能表达一种注意力分配。多头 = 多组独立的 softmax 权重 = 同时从多个角度看问题。比如头0关注最近的字母，头1关注开头字母，头2关注元音模式，头3关注名字长度。

### 第5步：注意力计算（L172-175）

**5a. 算相似度（L172）**

```python
attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
               for t in range(len(k_h))]
```

Q 和每个历史 K 做点积，除以 √head_dim（=2.0）防止数值太大。

```
q_h 和 BOS的k_h 点积 / 2.0 → 0.36
q_h 和 j的k_h   点积 / 2.0 → 0.60
q_h 和 a的k_h   点积 / 2.0 → 0.66
attn_logits = [0.36, 0.60, 0.66]
```

**5b. 变成权重（L173）**

```python
attn_weights = softmax(attn_logits)
# [0.36, 0.60, 0.66] → [0.27, 0.34, 0.39]  加起来=1
```

**5c. 加权求和（L174）**

```python
head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
            for j in range(head_dim)]
```

用权重把历史字母的 V 混合起来。权重大的字母贡献多。

**拼接（L175）**

```python
x_attn.extend(head_out)  # 4个头各输出4维，拼成16维
```

**融合（L176）+ 残差连接（L177）**

```python
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # 线性层融合4个头
x = [a + b for a, b in zip(x, x_residual)]             # 加上原始x
```

4个头的结论通过 attn_wo 矩阵融合。残差连接保底：就算注意力没学到有用的东西，原始信息也不会丢。

### 第6步：MLP 块（L178-184）

```python
x_residual = x                                         # 存副本
x = rmsnorm(x)                                         # 归一化
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])        # 16维→64维（升维展开）
x = [xi.relu() ** 2 for xi in x]                       # 激活函数：负数归零，正数平方
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])        # 64维→16维（降维压缩）
x = [a + b for a, b in zip(x, x_residual)]             # 残差连接
```

注意力负责"从历史字母中收集信息"，MLP 负责"对收集到的信息做加工处理"。

**Q: 激活函数 relu² 会不会太粗暴？**
确实粗暴——负数直接归零，正数还平方。但平方让输出更"稀疏"：小正数（0.1→0.01）几乎也被消掉，只有大正数能活下来。稀疏 = 模型被迫用少量特征做决策，不容易过拟合。大模型用更平滑的激活函数（GELU、SiLU），这个小模型用 relu² 够了。

**Q: 为什么非线性操作很重要？**
没有非线性，多层 linear 叠在一起还是 linear（`w2 × w1 × x = W × x`），模型只能学直线关系。relu² 打破了线性——不同的输入激活不同的维度组合（64维，每个开或关，理论上 2^64 种组合），模型就能对不同情况做出不同反应。一句话：线性只能画直线，非线性让模型能画曲线。

### 第7步：输出层（L186）

```python
logits = linear(x, state_dict['lm_head'])  # 16维→27维
```

lm_head 是 27×16 的矩阵，每一行是一个字母的"特征模板"。x 和每一行做点积，点积越大 = 越像这个字母 = 概率越高。

---

## 七、训练循环（L189-227）

### 优化器初始化（L189-192）

```python
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params)  # 一阶动量（方向记忆）
v = [0.0] * len(params)  # 二阶动量（大小记忆）
```

### 每一步的流程（L196-227）

```
取一个名字 → 分词 → 逐位置跑模型 → 算loss → 反向传播 → Adam更新 → 重复500次
```

#### 准备数据（L199-201）

```python
doc = docs[step % len(docs)]                         # 取一个名字，如 "jan"
tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]    # → [0, 10, 1, 14, 0]
n = min(block_size, len(tokens) - 1)                  # n=4，4组输入→目标配对
```

```
位置0: 输入BOS → 目标j    （看到开头，应该预测j）
位置1: 输入j   → 目标a    （看到j，应该预测a）
位置2: 输入a   → 目标n    （看到a，应该预测n）
位置3: 输入n   → 目标BOS  （看到n，应该预测结束）
```

#### 正向传播 + 算 loss（L204-212）

```python
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)  # 跑模型，得到27个分数
    probs = softmax(logits)                         # 变成概率
    loss_t = -probs[target_id].log()                # 交叉熵损失
    losses.append(loss_t)
loss = (1 / n) * sum(losses)                        # 平均loss
```

`-log(概率)` 的特性：概率接近1 → loss接近0（猜对了），概率接近0 → loss趋向无穷大（猜错了）。

#### 反向传播（L215）

```python
loss.backward()
```

从 loss 出发，沿计算图反向传播，算出每个参数的梯度。

#### Adam 优化器更新（L218-225）

```python
lr_t = learning_rate * (1 - step / num_steps)  # 学习率线性衰减
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad          # 方向记忆（惯性）
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2      # 大小记忆（自适应步长）
    m_hat = m[i] / (1 - beta1 ** (step + 1))              # 偏差修正
    v_hat = v[i] / (1 - beta2 ** (step + 1))              # 偏差修正
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)     # 更新参数
    p.grad = 0                                              # 清零梯度
```

Adam 相比普通梯度下降的优势：
- **m（一阶动量）**：如果最近几步梯度方向一致，加速前进（惯性）
- **v（二阶动量）**：梯度大的参数走慢点，梯度小的走快点（自适应）
- **学习率衰减**：开始大步走快速接近目标，后期小步走精细调整

---

## 八、推理（L229-243）

```python
temperature = 0.6
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS                                    # 从BOS开始
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)  # 跑模型
        probs = softmax([l / temperature for l in logits])  # temperature缩放+softmax
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:                            # 遇到BOS就结束
            break
        print(itos[token_id], end="")                  # 打印字母
```

和训练的区别：
- 不算 loss，不反向传播，不更新权重
- 用 temperature 控制"创造力"：小→保守（几乎只选概率最大的），大→随机
- 按概率随机采样，不是每次选最大的，所以每次生成的名字不同

---

## 九、关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| Value | 包装数字的节点，记录计算过程，支持自动求导 |
| 梯度（grad） | 参数增大一点，loss 怎么变 |
| 前向传播 | 输入→经过模型→得到输出 |
| 反向传播 | 从 loss 出发，沿计算图反向算每个参数的梯度 |
| loss | 模型猜错的程度，越小越好 |
| linear | 矩阵乘法，信息变换的基本工具 |
| softmax | 把分数变成概率（软选择） |
| rmsnorm | 归一化，拉齐数值范围 |
| 嵌入（embedding） | 查表，把编号变成向量 |
| Q/K/V | Q和K算相似度，V携带信息，三者独立生成 |
| 多头注意力 | 切成多份，从多个角度看问题 |
| 残差连接 | 输出 = 处理结果 + 原始输入，保底不丢信息 |
| MLP | 升维→激活→降维，对信息做非线性加工 |
| relu² | 负数归零，正数平方，制造稀疏性 |
| Adam | 带惯性和自适应步长的梯度下降 |
| temperature | 控制生成的随机程度 |
| lm_head | 输出层矩阵，每行是一个字母的特征模板 |
| 模型权重 | 训练学到的所有矩阵里的数字，存在模型文件里 |

