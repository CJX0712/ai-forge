# 架构说明 · AI Forge / 星锻

> 十层分解架构（L0–L10），每层都是可被不变量验证的「物理定律」。

## 全景

```
L0  数值内核    numeric.js      erf/logsumexp/Kahan/Welford/lgamma
L1  线性代数    linalg.js        GEMM(blocked)/Cholesky/Jacobi/SVD/Newton–Schulz
L1  随机        random.js        sfc32(确定性) / Vose alias
L1  自动微分    autodiff.js       tape 反向模式 + 中心差分 gradCheck
L2  优化器      optim.js          AdamW / SGD / Muon / L-BFGS
L3  网络模块    nn/modules.js     RMSNorm / RoPE / SwiGLU / MoE
L4  注意力      nn/attention.js    FlashAttention / GQA / MLA / KV-cache / 滑动窗口 / 线性注意力
L5  Transformer nn/transformer.js pre-norm, tied head, GQA, 可选 MoE
L6  经典 ML     ml/*              linear / svm / tree / cluster / instance / pca
L7  推理采样    infer/sampling.js top-k / top-p / min-p / beam / 投机解码
L8  验证        verify/assert.js  52 条可执行不变量
L9  基准        bench/run.mjs     复杂度声明 + 诚实基线对照
L10 单文件      tools/build-html 零依赖打包 → dist/ai-forge.html
```

数据流（训练一条样本）：
```
X ─▶ [L1 autodiff tape] ─▶ matmul/add/tanh ─▶ [L3 RMSNorm/RoPE/SwiGLU]
   ─▶ [L4 FlashAttention] ─▶ [L5 TransformerLM] ─▶ logits
   ─▶ crossEntropy ─▶ [L1 backward] ─▶ grads ─▶ [L2 Muon/AdamW] ─▶ 参数更新
```

## 关键设计决策与权衡

### 1. 自动微分：tape 而非图编译
**决策**：采用录制式 tape 反向模式。
**理由**：项目硬约束是「可验证性」。tape 让「梯度 vs 中心差分」的 `gradCheck` 变得平凡且精确——这是 PyTorch `gradcheck`、JAX `check_grads` 的黄金标准。
**权衡**：tape 比静态图编译略慢（每次前向重新录制），但对一个可验证、可读、零依赖的库，可读性 > 极限吞吐。吞吐瓶颈由 L9 基准显式标注，且后端可插拔（WASM/WebGPU）。

### 2. Muon：牛顿–舒尔茨正交化
**决策**：2-D 权重走 `SGD+Nesterov 动量 → Newton–Schulz 5 步正交化 → RMS 重缩放`；1-D 参数/embedding/head 回退 AdamW。
**数学**：Newton–Schulz 迭代 `X ← aX + bXAXᵀ + c(XAXᵀ)²X`（系数 3.4445, −4.7750, 2.0315）将谱半径映射到 [0.25, 1.6]，5 步即收敛。
**不变量**：`frobenius(O) ∈ [0.25·√n, 1.6·√n]` 且 `OᵀO ≈ I`（正交化验证）。

### 3. FlashAttention：在线 softmax
**决策**：流式计算注意力，维护 running max/m 与 numerators，O(1) 额外显存。
**不变量**：`max|flash − full| = 0`（机器精度内 bit-exact）；峰值显存 `T·dk` vs 全注意力的 `T²`。
**权衡**：因果掩码 + 分块（blockSize=32）。数值稳定性由 logsumexp 保证。

### 4. GQA / MLA：KV 压缩
- **GQA**（Ainslie 2023）：多 query 头共享少量 KV 头，降低解码显存与延迟。
- **MLA**（DeepSeek-V2 2024）：低秩 KV 潜变量；`KV cache = d_c·d_k`（本实现 2048 → 512 floats/token，4× 更小）。

### 5. 经典 ML：每个算法都是教科书标准实现
- SMO-SVM：Platt 1998，含 `Σαy=0` 与 KKT 互补松弛不变量（误差 2.2e-16）。
- GMM-EM：logsumexp 责任度，LL 单调↑不变量。
- k-means++：种子初始化，SSE 单调↓不变量。
- KD-tree：最大堆 k-近邻，与暴力结果 bit-exact。
- PCA：Jacobi 特征分解，Eckart–Young 误差 = 丢弃特征值之和（1e-9）。

### 6. 确定性
所有随机源为 **sfc32**（非 Math.random），由字符串种子哈希驱动。同一 seed → 同一序列 → 不变量可逐位复现（Transformer 确定性不变量 max|Δ|=0）。

### 7. 零依赖打包（build-html）
`tools/build-html.mjs` 是一个手写 mini-bundler，仅用 Node 内置模块，把 18 个 ES 模块解析为单一浏览器 IIFE：
- 处理 `import {a as b}` / `import * as ns` / `export *` / `export * as ns` / `export {..} from` 的优先级（局部 > 具名再导出 > 星号再导出，与 Node ESM 一致）。
- 字号 130KB，零运行时依赖、零外部请求——双击即跑。

## 模块职责一览

| 模块 | 职责 | 关键不变量 |
|---|---|---|
| numeric | 数值稳定基元 | Kahan 误差=0；erf 精度 1e-12 |
| linalg | 矩阵运算 | Newton–Schulz 正交；Cholesky 重构误差 |
| autodiff | 梯度 | gradCheck maxRel < 1e-5 |
| optim | 优化 | L-BFGS ‖∇f‖<1e-10；Muon 收敛 ≤2% 初值 |
| modules | NN 层 | RMSNorm ‖y‖=1；RoPE 仅依赖相对位置 |
| attention | 注意力 | Flash≡Full bit-exact；MLA 4× KV；windowed≡full@window≥T；linear direct≡recurrent |
| transformer | LM | 确定性；flash/full 一致 |
| ml/* | 经典 ML | SSE/LL 单调；KKT；Eckart–Young |
| sampling | 解码 | top-p 保质量；投机解码无损 |
| verify | 不变量 | 52/52 |

---
署名：晨星（Chenxing）· MIT
