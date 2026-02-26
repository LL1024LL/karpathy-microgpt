"""
学习 __add__ 的可运行示例
"""

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label  # 加个名字方便打印

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f'{self.label}=Value(data={self.data}, grad={self.grad})'


# ========== 场景1: 最简单的加法 ==========
print("=== 场景1: c = a + b ===")
a = Value(3.0, label='a')
b = Value(2.0, label='b')
c = a + b
c.label = 'c'

print(f"正向计算: a={a.data} + b={b.data} = c={c.data}")
print(f"c 的父节点: {[v.label for v in c._prev]}")
print(f"c 的运算符: {c._op}")

# 模拟反向传播
c.grad = 1.0  # 假设 c 就是 loss
c._backward()
print(f"反向传播后: a.grad={a.grad}, b.grad={b.grad}")
print(f"含义: a 增大1, c 增大{a.grad}; b 增大1, c 增大{b.grad}")
print()


# ========== 场景2: 链式加法 ==========
print("=== 场景2: d = a + b + c (链式) ===")
a = Value(1.0, label='a')
b = Value(2.0, label='b')
c = Value(3.0, label='c')
tmp = a + b       # tmp = 3.0
tmp.label = 'tmp'
d = tmp + c       # d = 6.0
d.label = 'd'

print(f"正向: a={a.data} + b={b.data} = tmp={tmp.data}")
print(f"正向: tmp={tmp.data} + c={c.data} = d={d.data}")

# 反向传播: 从 d 开始,一步步往回传
d.grad = 1.0
d._backward()     # 梯度传给 tmp 和 c
print(f"d._backward() 后: tmp.grad={tmp.grad}, c.grad={c.grad}")

tmp._backward()   # 梯度从 tmp 传给 a 和 b
print(f"tmp._backward() 后: a.grad={a.grad}, b.grad={b.grad}")
print(f"最终: a,b,c 的梯度都是 {a.grad}, 因为每个变量增大1, d 都增大1")
print()


# ========== 场景3: 同一个变量用两次 ==========
print("=== 场景3: c = a + a (同一个变量用两次) ===")
a = Value(5.0, label='a')
c = a + a
c.label = 'c'

print(f"正向: a={a.data} + a={a.data} = c={c.data}")

c.grad = 1.0
c._backward()
print(f"反向后: a.grad={a.grad}")
print(f"含义: a 增大1, c 增大{a.grad} (因为 a 出现了两次, 梯度累加)")
print()


# ========== 场景4: Value + 普通数字 ==========
print("=== 场景4: c = a + 10 (Value + 普通数字) ===")
a = Value(3.0, label='a')
c = a + 10  # 10 会被自动包装成 Value(10)
c.label = 'c'

print(f"正向: a={a.data} + 10 = c={c.data}")

c.grad = 1.0
c._backward()
print(f"反向后: a.grad={a.grad}")
print(f"含义: 加一个常数不影响梯度, a 的梯度还是 {a.grad}")
print()


# ========== 场景5: 乘法 vs 加法 ==========
print("=== 场景5: 乘法 c = a * b ===")

class ValueMul(Value):
    def __mul__(self, other):
        other = other if isinstance(other, ValueMul) else ValueMul(other)
        out = ValueMul(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

a = ValueMul(3.0, label='a')
b = ValueMul(2.0, label='b')
c = a * b
c.label = 'c'

print(f"正向: a={a.data} * b={b.data} = c={c.data}")

c.grad = 1.0
c._backward()
print(f"反向后: a.grad={a.grad}, b.grad={b.grad}")
print(f"含义: a.grad={a.grad} 等于 b 的值, b.grad={b.grad} 等于 a 的值")
print()

# 对比: 如果 b 很大呢?
print("=== 场景6: b 很大时, a 的梯度也很大 ===")
a = ValueMul(3.0, label='a')
b = ValueMul(100.0, label='b')
c = a * b
c.label = 'c'

print(f"正向: a={a.data} * b={b.data} = c={c.data}")

c.grad = 1.0
c._backward()
print(f"反向后: a.grad={a.grad}, b.grad={b.grad}")
print(f"含义: b=100 很大, 所以 a 的微小变化会被放大100倍")
