# LLaMA2 推理源码解读

> 基于 llama2.npy（573行，3个文件），纯 NumPy 实现的 LLaMA2 推理。
> 本报告以 gpt.py 为参照，重点讲解 LLaMA 架构相对于 GPT 的演进。

---

## 一、整体架构

### 文件结构

```
llama2.npy (573行，3个文件)
├── tokenizer_npy.py (91行)   分词器：BPE子词分词
├── model_npy.py (349行)      模型：Transformer前向传播
└── run_npy.py (133行)        入口：加载权重 + 推理
```

### 与 gpt.py 的对应关系

```
llama2.npy                          gpt.py
─────────────────────────────────────────────────
tokenizer_npy.py                    stoi/itos 字符级分词
  Tokenizer: BPE分词器

model_npy.py
  RMSNorm                           rmsnorm()
  NumpyLinear                        linear()
  NumpyEmbedding                     wte/wpe 查表
  softmax / silu / sigmoid           softmax / relu²
  precompute_freqs_cis + apply_rotary   wpe 位置嵌入
  repeat_kv                          （无，gpt.py是标准MHA）
  Attention                          gpt()里的注意力部分
  FeedForward                        gpt()里的MLP部分
  TransformerBlock                   gpt()里的一层循环
  Transformer                        gpt() + 推理循环

run_npy.py
  加载权重 + 分词 + 推理              L189-243 训练+推理
```

### 关键差异一览

| 组件 | gpt.py | llama2.npy | 差异原因 |
|---|---|---|---|
| 位置编码 | wpe 查表（学习式） | RoPE 旋转（计算式） | 不受长度限制 |
| 注意力 | 标准 MHA | 支持 GQA | 省 KV Cache |
| 激活函数 | relu² | SwiGLU（silu + 门控） | 更平滑灵活 |
| MLP | 2个矩阵(fc1,fc2) | 3个矩阵(w1,w2,w3) | SwiGLU 需要门控矩阵 |
| 分词器 | 字符级（27个token） | BPE（32000个token） | 更少token表达更多文本 |
| KV Cache | 有（keys/values列表） | 无（每次重算） | 简化实现 |
| 自动求导 | Value 类 | 无 | 只做推理不训练 |
| 输出层 | 独立 lm_head | 复用嵌入表 | 省参数 |

---

## 二、BPE 分词器（tokenizer_npy.py）

### 与 gpt.py 字符级分词的对比

```
gpt.py（字符级）:
  "hello" → [8, 5, 12, 12, 15]  5个token，每个字母一个
  词汇表: 27个（BOS + a~z）

BPE:
  "hello" → 可能1~2个token（常见词被合并）
  词汇表: 32000个
```

### BPE 编码流程

```python
def encode(self, initial_string, bos, eos):
    # 第1步：拆成单个字符
    tokens = [self.vocab2index[char] for char in initial_string]

    # 第2步：反复合并相邻的token对
    while True:
        # 遍历所有相邻对，找分数最高的
        for i in range(len(tokens) - 1):
            string = self.index2vocab[tokens[i]] + self.index2vocab[tokens[i + 1]]
            str_id = self.vocab2index.get(string, None)
            if str_id is not None and score > best_score:
                best_score, best_idx = score, i
        # 合并最佳对，直到没有可合并的
        if best_idx == -1: break
        tokens[best_idx] = best_id
        tokens.pop(best_idx + 1)
```

示例：

```
"hello" 编码过程:
  拆字符: ['h', 'e', 'l', 'l', 'o']
  合并 ll → [h, e, ll, o]
  合并 ell → [h, ell, o]
  合并 hell → [hell, o]
  合并 hello → [hello]（如果词汇表里有）
```

### 为什么用 BPE 不用字符级

- token 数量少 → 序列短 → 注意力 O(N²) 计算量大幅减少
- KV Cache 更小 → 能服务更多用户
- 同样的 block_size 能覆盖更长的文本

嵌入表虽然从 27×16 变成了 32000×4096，但嵌入操作是查表（零计算），在整个模型中占比极小（<2%）。

---

## 三、RoPE 位置编码

### gpt.py 的查表式 vs RoPE 的旋转式

```
gpt.py（查表）:
  位置0 → 查表第0行 → 得到向量 → 加到嵌入上
  位置7 → 查表第7行
  位置8 → 没有第8行，崩了
  受限于表的大小

RoPE（旋转）:
  位置0 → 不转
  位置1 → 转10°
  位置2 → 转20°
  位置99999 → 转999990°
  给任何位置号都能算，不受限制
```

### 核心原理

每个位置的 Q 和 K 被旋转不同的角度。做点积时，结果只取决于两个位置的角度差（相对距离）：

```
位置3的Q（转30°）· 位置1的K（转10°）→ 角度差20°
位置5的Q（转50°）· 位置3的K（转30°）→ 角度差20°
同样的距离 = 同样的效果
```

### 多频率：钟表比喻

128维向量有64对维度，每对用不同的旋转速度：

```
维度对0（秒针）: 转得快 → 区分近距离
维度对1（分针）: 转得慢 → 区分中距离
...
维度对63（时针）: 转得极慢 → 区分远距离
```

单个维度对会"转回来"（超过360°），但64对组合在一起，每个位置的"指纹"唯一。

### 代码实现

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    # 算每对维度的旋转速度（越往后越慢）
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    # 算每个位置的旋转角度
    t = np.arange(end)
    freqs = np.outer(t, freqs)
    # 取cos和sin备用
    return np.cos(freqs), np.sin(freqs)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # 每两个相邻维度配对，用对应角度旋转
    xq_r, xq_i = xq[..., 0::2], xq[..., 1::2]
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    # K 同理，V 不旋转
```

### 主流位置编码对比

| 方式 | 加在哪 | 长度限制 | 现状 |
|---|---|---|---|
| 正弦编码 | 加到嵌入上 | 无 | 已淘汰 |
| 学习式（gpt.py） | 加到嵌入上 | 受表大小限制 | 小模型用 |
| RoPE | 旋转Q和K | 无 | 主流 |
| ALiBi | 减注意力分数 | 无 | 少数用 |

### 硬件友好性

RoPE 只需要乘加运算，计算量占比 ~0.005%，对任何硬件零要求。

---

## 四、GQA——分组查询注意力

### 与 gpt.py 标准 MHA 的对比

```
gpt.py（MHA）: Q 4头, K 4头, V 4头 → 每个Q头有专属KV
llama2.npy（GQA）: Q 32头, K 8头, V 8头 → 每4个Q头共享一组KV

Q头0,1,2,3   → 共享 K头0, V头0
Q头4,5,6,7   → 共享 K头1, V头1
...
Q头28,29,30,31 → 共享 K头7, V头7
```

### 代码实现

```python
class Attention:
    def __init__(self, args):
        self.n_heads = args.n_heads          # Q头数: 32
        self.n_kv_heads = args.n_kv_heads    # KV头数: 8
        self.n_rep = self.n_heads // self.n_kv_heads  # 重复倍数: 4

        self.wq = NumpyLinear(dim, n_heads * head_dim)      # 生成32个Q头
        self.wk = NumpyLinear(dim, n_kv_heads * head_dim)   # 只生成8个K头
        self.wv = NumpyLinear(dim, n_kv_heads * head_dim)   # 只生成8个V头
```

`repeat_kv` 把8个KV头复制4遍变成32个，和Q对齐后做点积：

```python
def repeat_kv(x, n_rep):
    # [batch, seq, 8, head_dim] → [batch, seq, 32, head_dim]
    x = np.expand_dims(x, axis=3)
    x = np.tile(x, (1, 1, 1, n_rep, 1))
    return x.reshape(...)
```

### 为什么共享KV质量不受影响

- Q 的多样性是主要的——不同的Q问不同的问题，从同一套KV里得到不同的答案
- 实验证明KV头之间本身有大量冗余
- Google 论文实测：32个KV头砍到8个，质量只掉 ~0.5%

### 显存收益

```
MHA (32个KV头): 64 MB/层
GQA (8个KV头):  16 MB/层  ← 省4倍KV Cache
```

---

## 五、SwiGLU——门控MLP

### 与 gpt.py relu² MLP 的对比

```python
# gpt.py: 2个矩阵，一条路
x = linear(x, fc1)              # 升维 16→64
x = [xi.relu() ** 2 for xi in x] # 激活：负数归零，正数平方
x = linear(x, fc2)              # 降维 64→16

# llama2.npy: 3个矩阵，两条路
h1 = silu(self.w1.forward(x))   # 路1：升维 + silu激活（门）
h3 = self.w3.forward(x)         # 路2：升维（信息）
h = h1 * h3                     # 门控：逐元素相乘
out = self.w2.forward(h)        # 降维
```

### silu 激活函数

```python
def silu(x):
    return x * sigmoid(x)    # sigmoid(x) = 1/(1+e^(-x))
```

比 relu 平滑，梯度不会突然断掉。

### 门控机制的优势

```
relu²（无门控）:
  负数直接归零 → 信息丢失，一刀切

SwiGLU（有门控）:
  w1+silu = 门卫，决定"哪些信息通过"（值在0~1附近）
  w3 = 信使，携带"要传递的信息"
  门卫和信使各自独立学习，更灵活
  负数信息也能通过（门开着就行）
```

代价是多一个矩阵（+50%参数），但效果提升明显，所有主流大模型都在用。

---

## 六、权重加载（run_npy.py）

### 文件结构

```
.bin 文件 = [文件头 28字节] + [一长串浮点数]

文件头（7个整数）:
  dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len

权重数据（按固定顺序排列）:
  嵌入表 → 每层的norm → 每层的wq → 每层的wk → 每层的wv → 每层的wo
  → 每层的ffn_norm → 每层的w1 → 每层的w2 → 每层的w3
  → 最终norm → RoPE的cos/sin表
```

### 加载过程

```python
def deserialize_np(t, f):
    num_elements = t.size
    data = struct.unpack(f'{num_elements}f', f.read(4 * num_elements))
    return np.array(data).reshape(t.shape)
```

从文件里读一串浮点数，reshape 成矩阵。每个浮点数占4字节（FP32）。

### 输出层复用嵌入表

```python
transformer.output.weight = transformer.tok_embeddings.weight
```

嵌入（编号→向量）和输出（向量→分数）是互逆操作，共享同一个矩阵，省一整个 [vocab_size × dim] 的参数。

### 真实模型的权重格式

| 格式 | 特点 | 现状 |
|---|---|---|
| safetensors | 安全、支持mmap、按需加载 | 当前主流 |
| PyTorch .bin | pickle序列化，有安全风险 | 旧格式 |
| GGUF | 支持量化，CPU推理专用 | llama.cpp 用 |

---

## 七、generate() 推理循环

### 与 gpt.py 的核心区别

```
gpt.py（有KV Cache）:
  每次送1个token，历史K/V从Cache取
  生成10个token: 计算量 = 1+1+1+...+1 = 10次

llama2.npy（无KV Cache）:
  每次送整个序列，重算所有token
  生成10个token: 计算量 = 1+2+3+...+10 = 55次
```

### 推理流程

```python
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        logits = self.forward(index)          # 整个序列送入
        logits = logits[:, -1, :]             # 只取最后位置的输出
        logits = logits / temperature          # temperature缩放
        probs = softmax(logits)                # 变成概率
        idx_next = sample(probs)               # 采样
        index = np.concatenate((index, idx_next), axis=1)  # 拼到末尾
```

`logits[:, -1, :]`：模型对每个位置都输出了预测，但只有最后一个位置是"预测下一个token"，前面位置的预测已经知道答案了。

### temperature 控制

```
temperature < 1: 差距放大，更确定，倾向选概率最大的
temperature = 1: 原始分布
temperature > 1: 差距缩小，更随机
temperature = 0: 贪心解码，永远选最大的
```

---

## 八、Decoder-Only 架构

gpt.py 和 llama2.npy 都是 Decoder-Only 架构。

### 三种 Transformer 架构

| 架构 | 能看到什么 | 适合任务 | 代表 |
|---|---|---|---|
| Encoder-Only | 所有token（前后都看） | 理解（分类、填空） | BERT |
| Decoder-Only | 只看前面（因果掩码） | 生成（对话、写代码） | GPT、LLaMA、Qwen |
| Encoder-Decoder | Encoder全看，Decoder看前面 | 翻译、摘要 | T5 |

### 为什么 Decoder-Only 成为主流

- 生成能力是刚需（对话、写代码、写文章）
- 结构简单，容易扩大规模
- 模型够大后，理解能力也够用
- 训练效率高（每个位置都能算loss）

---

## 九、关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| BPE | 反复合并高频相邻对，用更少token表达更多文本 |
| RoPE | 用旋转角度编码位置，不受长度限制，天然编码相对距离 |
| GQA | 多个Q头共享一组KV，省KV Cache显存 |
| repeat_kv | 把少量KV头复制多份，和Q头数对齐 |
| SwiGLU | 两条路：一条当门（silu），一条传信息，逐元素相乘 |
| silu | x × sigmoid(x)，比relu平滑 |
| 门控 | 一路控制"通过多少"，一路携带"什么信息" |
| safetensors | 安全的模型权重格式，支持内存映射 |
| 权重共享 | 嵌入表和输出层用同一个矩阵，省参数 |
| Decoder-Only | 只能看前面不能看后面，天然适合生成任务 |
| temperature | 控制生成随机程度，小→确定，大→随机 |
| KV Cache | 存历史K/V避免重算（llama2.npy未实现） |
