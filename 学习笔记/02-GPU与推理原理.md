# GPU与大模型推理原理

> 从硬件特性到推理引擎，理解大模型推理的完整链路。
> 本报告结合 gpt.py 的代码逻辑，从推理视角展开讲解。

---

## 一、为什么需要GPU？——从矩阵乘法说起

### 推理的核心操作

gpt.py 的 `linear` 函数就是矩阵乘法：

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

在 gpt.py 中，x 是16维，w 是16×16，一次 linear 做 256 次乘加。但在真实大模型（如 LLaMA-70B）中：
- x 是 8192 维，w 是 8192×8192
- 一次 linear = **6700万次**乘加
- 一次前向传播要跑几十次 linear，总计**几百亿次**运算

### CPU vs GPU 的架构差异

```
CPU（如 i9）:                      GPU（如 A100）:
┌──────────────────┐              ┌──────────────────────────┐
│  ██  ██  ██  ██  │              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ │
│  ██  ██  ██  ██  │              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ │
│  (8-24个大核)     │              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ │
│  每个核很强       │              │ (6912个小核)              │
│  擅长复杂逻辑     │              │ 每个核很弱                │
└──────────────────┘              │ 但能同时干活              │
                                  └──────────────────────────┘
```

### 为什么矩阵乘法天然适合GPU

矩阵乘法的本质：w 有多少行，就做多少次点积。**每一行的点积互相独立**。

```
w 的第0行和 x 做点积 → 输出第0个值    ← 核心0算
w 的第1行和 x 做点积 → 输出第1个值    ← 核心1算
...
w 的第8191行和 x 做点积 → 输出第8191个值 ← 核心8191算
```

| | CPU (i9-13900K) | GPU (A100) |
|---|---|---|
| 核心数 | 24 | 6912 |
| 浮点算力 | ~1 TFLOPS | ~312 TFLOPS |
| 显存/内存带宽 | ~90 GB/s | ~2000 GB/s |

GPU 理论算力是 CPU 的 **300倍**，带宽是 **22倍**。

---

## 二、GPU的内存层级——推理的真正瓶颈

### 两级存储

```
GPU 芯片
┌─────────────────────────────────────────┐
│   ┌───────────┐                         │
│   │   SRAM    │  ← 片上缓存，20MB       │
│   │  (快,小)   │    带宽: ~19 TB/s       │
│   └─────┬─────┘                         │
│         │  搬数据（瓶颈在这）             │
│   ┌─────┴─────┐                         │
│   │    HBM    │  ← 显存，80GB           │
│   │  (慢,大)   │    带宽: 2 TB/s         │
│   └───────────┘                         │
│   计算单元: 6912个核，312 TFLOPS         │
└─────────────────────────────────────────┘
```

### 搬数据 vs 算数据

以 LLaMA-70B 的一个 linear 层（8192×8192，FP16）为例：

```
搬数据: 128MB / 2TB/s = 0.064 ms
算数据: 1.34亿次 / 312T = 0.00043 ms

搬运时间是计算时间的 150倍
```

GPU 的计算核心在 99% 的时间里都在等数据从显存搬过来。

### 两种瓶颈类型

| | 计算密集 (Compute-bound) | 访存密集 (Memory-bound) |
|---|---|---|
| 瓶颈 | 算力不够 | 带宽不够 |
| 典型场景 | 训练、Prefill阶段 | Decode 逐token生成 |
| GPU利用率 | 高 | 低 |

Batching 的作用：权重只搬一次，算多份，摊薄搬运成本。

```
batch=1:   搬128MB权重，算1次   → 访存密集，GPU闲着
batch=128: 搬128MB权重，算128次 → 计算密集，GPU跑满
```

### 为什么只看带宽不看延迟

HBM 延迟约 100~200 纳秒（第一个字节到达的时间）。搬 128MB 权重时：

```
延迟部分: 0.0002 ms（占 0.3%）
传输部分: 0.064 ms（占 99.7%）
```

数据量大时延迟可忽略。GPU 还通过大量线程切换来隐藏延迟——当一组线程等数据时，立刻切换到另一组去算。

---

## 三、底层算子优化——让GPU别闲着

### 什么是算子（Kernel）

算子就是 GPU 上执行的一个函数，对应 gpt.py 的每个函数：

```
gpt.py 函数          →  GPU 算子
linear(x, w)         →  MatMul kernel
softmax(logits)      →  Softmax kernel
rmsnorm(x)           →  RMSNorm kernel
xi.relu() ** 2       →  Activation kernel
```

每个算子独立执行：从 HBM 读数据 → 搬到 SRAM → 计算 → 结果写回 HBM。

### 算子融合（Kernel Fusion）

多个小算子合并成一个大算子，中间结果留在 SRAM 不写回 HBM：

```
融合前（逐元素操作）:          融合后:
  读x → relu → 写回             读x → relu → 平方 → 写回
  读x → 平方 → 写回             （省掉一次读写）
  （2次读写）                    （1次读写）
```

### FlashAttention——注意力的专项优化

朴素注意力的问题：要把巨大的注意力矩阵（序列长4096时为 4096×4096）写到 HBM。

FlashAttention 的做法：分块在 SRAM 里算完，注意力矩阵从头到尾不出现在 HBM 里。

```
朴素注意力:                      FlashAttention:
  Q×K^T → 写HBM                   把K,V分成小块
  读回 → softmax → 写HBM          每块在SRAM里算完
  读回 → ×V → 写HBM               累加到结果
  （3次HBM往返）                   （只写最终结果到HBM）
```

效果：快 2~4 倍，显存从 O(N²) 降到 O(N)。

### 量化（Quantization）

让权重本身变小，搬得更快：

| 精度 | 每个权重 | LLaMA-70B 模型大小 |
|---|---|---|
| FP16 | 2 字节 | 140 GB |
| INT8 | 1 字节 | 70 GB |
| INT4 | 0.5 字节 | 35 GB |

代价是精度损失，但实践中 INT4 量化后质量下降很小。

---

## 四、注意力机制变体总览

### 按解决的问题分类

| 方法 | 解决什么问题 | 核心思路 | 代表使用者 |
|---|---|---|---|
| FlashAttention | HBM带宽瓶颈 | 分块在SRAM算，不落地HBM | 几乎所有引擎 |
| PagedAttention | KV Cache显存碎片 | 借鉴OS分页，动态分配 | vLLM |
| GQA | KV Cache太大 | 多头共享KV，省显存 | LLaMA-2/3, Gemma |
| MQA | KV Cache太大 | 所有头共享一份KV | PaLM, StarCoder |
| MLA | KV Cache太大 | 压缩KV到低维潜变量 | DeepSeek-V2/V3 |
| Sliding Window | 长序列O(N²) | 只看最近W个token | Mistral |
| Sparse Attention | 长序列O(N²) | 跳着看，稀疏模式 | GPT-3, Longformer |
| Ring Attention | 超长序列跨卡 | 多卡环形传递KV块 | 研究阶段 |

它们不互斥，实际推理引擎会组合使用。

---

## 五、KV Cache——推理引擎的核心优化

### gpt.py 中的 KV Cache

```python
keys[li].append(k)      # 每处理一个token，存K
values[li].append(v)     # 存V
```

### 为什么需要缓存

不缓存时，生成 N 个 token 要算 N×(N+1)/2 次 linear；有缓存只要 N 次：

```
N=100:   不缓存 5050次  vs  缓存 100次   → 省 50倍
N=1000:  不缓存 50万次  vs  缓存 1000次  → 省 500倍
N=4096:  不缓存 838万次 vs  缓存 4096次  → 省 2048倍
```

### 显存代价

以 LLaMA-70B（GQA，80层）为例：

```
每个token每层: 4 KB
80层: 320 KB / token
一条序列4096 token: 1.28 GB
同时服务32个用户: 41 GB  ← 光KV Cache就占半张A100
```

### PagedAttention——解决碎片

借鉴操作系统虚拟内存分页，按小块动态分配，显存利用率从 ~50% 提升到 ~95%。

```
朴素分配:                          PagedAttention:
  预分配固定长度，大量空洞            按页动态分配，紧凑排列
  利用率 ~50%                       利用率 ~95%
```

### GQA/MQA/MLA——让 Cache 变小

```
MHA:  32头 × 128维 × 4096长 × 2(KV) × 2字节 = 64 MB/层
GQA:  8组  × 128维 × 4096长 × 2(KV) × 2字节 = 16 MB/层  ← 省4倍
MQA:  1份  × 128维 × 4096长 × 2(KV) × 2字节 = 2 MB/层   ← 省32倍
```

---

## 六、Prefill vs Decode——推理的两个阶段

### 两阶段对比

| | Prefill | Decode |
|---|---|---|
| 一次处理 | N 个 token（整个 prompt） | 1 个 token |
| 瓶颈类型 | 计算密集（GPU跑满） | 访存密集（GPU空闲） |
| GPU利用率 | 高（60~80%） | 低（1~5%） |

### gpt.py 没有 Prefill

gpt.py 始终是 decode 模式——每次只处理一个 token：

```python
for pos_id in range(block_size):
    logits = gpt(token_id, pos_id, keys, values)  # 每次喂1个token
```

真实引擎的 Prefill 把已知 prompt 打包成矩阵一次性处理，数学上等价但效率高得多。

### 因果掩码（Causal Mask）

gpt.py 串行处理，天然看不到未来（K 还没 append）。Prefill 并行处理时所有 K 同时存在，需要掩码遮住未来：

```
加掩码后的注意力矩阵:
         K_BOS  K_j   K_a   K_n
Q_BOS [  0.8   -∞    -∞    -∞  ]   ← 只看自己
Q_j   [  0.4    0.7  -∞    -∞  ]   ← 看BOS和自己
Q_a   [  0.2    0.5   0.8  -∞  ]   ← 看BOS、j、自己
Q_n   [  0.1    0.4   0.3   0.6 ]  ← 看所有
```

`-∞` 经过 softmax 后变成 0，等于完全看不见。"因果"含义：原因在前，结果在后，每个 token 只能看到之前的。

### Continuous Batching

Decode 阶段 GPU 利用率低，通过把多个用户的请求拼在一起处理来提高利用率：

```
batch=1:   GPU利用率 ~2%
batch=32:  GPU利用率 ~60%
batch=64:  GPU利用率 ~90%
```

有人结束就立刻塞新请求进来，GPU 始终满载。

### 用户体感指标

```
TTFT (Time To First Token): 首字延迟，取决于 Prefill 速度
TPOT (Time Per Output Token): 每token延迟，取决于 Decode 速度
```

---

## 七、模型架构差异

### config.json 中的 architectures 字段

```json
"architectures": ["Qwen3_5ForConditionalGeneration"]
```

告诉推理引擎用哪个 Python 类来加载和运行模型。每个 architecture 对应一份前向传播代码。

### 主流架构的组件选择

| 组件 | Qwen3 | ChatGLM4 | LLaMA-3 | Mistral | DeepSeek-V3 |
|---|---|---|---|---|---|
| 归一化 | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| 激活函数 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| 位置编码 | RoPE | RoPE | RoPE | RoPE | RoPE |
| 注意力 | GQA | MQA | GQA | GQA+滑动窗口 | MLA |
| MLP | 门控MLP | 门控MLP | 门控MLP | 门控MLP | MoE |

骨架一样（norm → attention → norm → mlp 循环），具体组件不同。

---

## 八、推理引擎全景

### 引擎模块对应关系

```
推理引擎
├── 模型执行层
│   ├── 矩阵乘法 ← GPU并行点积
│   ├── FlashAttention ← 注意力不落地HBM
│   ├── 算子融合 ← 减少HBM读写
│   └── 量化 ← 权重变小搬得快
├── 显存管理层
│   ├── KV Cache ← 存历史K/V避免重算
│   ├── PagedAttention ← 分页管理减少碎片
│   └── GQA/MQA/MLA ← 让Cache更小
├── 调度层
│   ├── Prefill/Decode分离
│   └── Continuous Batching ← 动态拼批
└── 底层硬件
    ├── GPU算力（TFLOPS）
    ├── HBM带宽
    └── SRAM缓存
```

### MindIE vs vLLM-Ascend

| | MindIE | vLLM-Ascend |
|---|---|---|
| 新模型适配 | 2~4周（手写定制算子） | 0-day（通用算子组合） |
| 单模型性能 | 更高 | 较低 |
| 设计思路 | 深度绑定模型架构 | 模型逻辑与硬件解耦 |

经典的**性能 vs 灵活性**权衡。

### 主流推理引擎

| 引擎 | 核心贡献 | 适合场景 |
|---|---|---|
| vLLM | PagedAttention + Continuous Batching | 通用场景 |
| TensorRT-LLM | 深度算子优化，NVIDIA绑定 | 极致性能 |
| SGLang | RadixAttention（前缀Cache共享） | 多轮对话 |
| llama.cpp | CPU/边缘推理，极致量化 | 本地部署 |

---

## 九、关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| TFLOPS | 每秒万亿次浮点运算，衡量GPU算力 |
| HBM | 高带宽显存，存模型权重，带宽是瓶颈 |
| SRAM | 片上缓存，极快但很小（20MB） |
| 计算密集 | 瓶颈在算力，GPU跑满（Prefill阶段） |
| 访存密集 | 瓶颈在带宽，GPU空闲（Decode阶段） |
| 算子 | GPU上执行的一个函数（kernel） |
| 算子融合 | 合并多个算子，减少HBM读写 |
| FlashAttention | 注意力分块在SRAM算完，不落地HBM |
| PagedAttention | KV Cache按页动态分配，减少碎片 |
| KV Cache | 存历史token的K/V，避免重复计算 |
| GQA | 多头共享KV组，省KV Cache显存 |
| MLA | 压缩KV到低维潜变量，极致省显存 |
| 因果掩码 | 下三角矩阵，防止看到未来token |
| Prefill | 一次性处理整个prompt，计算密集 |
| Decode | 逐token生成，访存密集 |
| Continuous Batching | 动态拼批，GPU始终满载 |
| TTFT | 首字延迟，取决于Prefill速度 |
| TPOT | 每token延迟，取决于Decode速度 |
| 量化 | 用更少位数表示权重，省显存省带宽 |
