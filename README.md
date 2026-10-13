# AI Forge · 星锻

> 零依赖 · 全自验证 · 可复现的 AI 系统 —— 数值内核 / 自动微分 / Muon·L-BFGS / FlashAttention·GQA·MLA / 经典 ML 全谱系

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![node](https://img.shields.io/badge/node-%3E%3D18-green.svg)](https://nodejs.org)
[![invariants](https://img.shields.io/badge/executable%20invariants-52%2F52-brightgreen.svg)](./test)
[![zero-deps](https://img.shields.io/badge/runtime%20deps-0-lightgrey.svg)](./src)

**AI Forge（星锻）** 是一个用纯 JavaScript 编写、零运行时依赖、可由浏览器与 Node 22 直接运行的 AI 系统。
它不重复造轮子——每一层都建立在经过同行评审的世界级算法之上（Muon、FlashAttention、GQA/MLA、RoPE、L-BFGS、SMO、Jacobi 特征分解……），
并用**可执行不变量（executable invariants）**把每个模块的数学声明验证到机器精度。

---

## 一、为什么是它（技术选型理由）

| 设计目标 | 选型 | 零重写依据 |
|---|---|---|
| 数值稳定性 | Kahan 求和、logsumexp、Numerical-Recipes `erf/erfc/lgamma` | 浮点求和误差 1e-11 → 0；`erf(1)=0.8427007929` |
| 自动微分 | 反向模式 tape + 中心差分梯度检验 | Linnainmaa 1970 / Rumelhart 1986；`gradcheck maxRel=6e-9` |
| 2-D 权重优化 | **Muon**（Newton–Schulz 正交化 + Nesterov 动量） | Jordan 2024 / Moonshot 2025 |
| 通用优化 | AdamW（解耦权重衰减）+ L-BFGS（强 Wolfe 线搜索） | Loshchilov 2019 / Nocedal 1980 |
| 注意力 | **FlashAttention**（在线 softmax 流式）+ GQA + MLA 低秩 KV | Dao 2022 / Ainslie 2023 / DeepSeek-V2 2024 |
| 位置编码 | RoPE（仅依赖相对位置） | Su 2021 |
| 经典 ML | SMO-SVM / CART-RF-GBDT / k-means++ / GMM-EM / KD-tree / PCA | Platt 1998 / Breiman 1984 / Lloyd / Dempster 1977 |

**核心原则（三条）**
1. **零重写**：先检索并复用世界最优解，组合与超越，绝不从零造轮子。
2. **集大成**：每个模块都必须站在世界级结果之上；一旦发现更强的成熟方案，立即替换自写实现。
3. **可验证**：唯一交付标准是「可运行、可复现、可验证」—— 52 条可执行不变量、基准测试、确定性随机源（sfc32）全部对齐。

---

## 二、目录结构

```
ai-forge/
├─ src/
│  ├─ core/            L0–L2  数值内核 / 线性代数 / 随机 / 自动微分 / 优化器
│  │   ├─ numeric.js    erf, logsumexp, softmax, Kahan, Welford, lgamma ...
│  │   ├─ linalg.js     GEMM(blocked), Cholesky, Jacobi/SVD, Newton–Schulz, Muon
│  │   ├─ random.js     sfc32 PRNG (确定性), Vose alias 采样
│  │   ├─ autodiff.js   反向模式 tape, 中心差分 gradcheck
│  │   └─ optim.js      AdamW, SGD, Muon, L-BFGS
│  ├─ nn/              L3–L5  RMSNorm/RoPE/SwiGLU/MoE, FlashAttention/GQA/MLA, Transformer
│  │   ├─ modules.js     rmsnorm, rope, swiglu, moeRoute, initLinear
│  │   ├─ attention.js   attentionFull, flashAttention, KVCache, decodeStep, MLA
│  │   └─ transformer.js TransformerLM (pre-norm, tied head, GQA, 可选 MoE)
│  ├─ ml/              L6      经典 ML 全谱系
│  │   ├─ linear.js      OLS(QR), Ridge, LASSO(坐标下降), Logistic(IRLS)
│  │   ├─ svm.js         SMO SVM (KKT 互补松弛)
│  │   ├─ tree.js        CART / RandomForest / GradientBoosting
│  │   ├─ cluster.js      k-means++ / GMM-EM / agglomerative
│  │   ├─ instance.js     KD-tree / KNN / GaussianNB
│  │   ├─ pca.js         PCA(Eckart–Young) / kernel PCA
│  │   └─ stat.js         metrics, r2, mse, trainTestSplit
│  ├─ infer/           L7      采样：top-k/top-p/min-p, beam, 投机解码
│  │   └─ sampling.js     sampleLogits, beamSearch, speculateAccept, perplexity
│  └─ verify/          L8      可执行不变量框架
│      └─ assert.js      Invariant, approx, assertTrue, assertMonotone, timeIt
├─ test/              L8      52 条可执行不变量（单一命令全跑）
│  ├─ run.mjs          入口：node test/run.mjs
│  └─ invariants.mjs    全部不变量定义
├─ bench/             L9      量化基准（含诚实基线对照）
│  └─ run.mjs          node bench/run.mjs
├─ tools/             build-html.mjs  零依赖打包器 → 单文件 HTML
├─ bin/               forge.mjs        CLI 入口（demo/test/bench/selfcheck）
├─ dist/              ai-forge.html    单文件浏览器版（130KB，零依赖）
└─ docs/              ARCHITECTURE / BENCHMARKS / ROADMAP
```

---

## 三、运行与验证

```bash
# 0. 环境（无需 npm install —— 零依赖）
node --version            # 需要 >= 18

# 1. 可执行不变量自检（52 条全跑，退出码 0 = 全绿）
npm test                 # 等价 node test/run.mjs

# 2. 量化基准
npm run bench            # 等价 node bench/run.mjs

# 3. 端到端 demo（梯度检验 + Transformer 等价性 + 贪心生成）
npm run demo             # 等价 node bin/forge.mjs demo

# 4. 生成单文件浏览器版（dist/ai-forge.html，可直接双击打开）
npm run build            # 等价 node tools/build-html.mjs

# 5. CI 门禁（test + bench 全绿才退出 0）
node bin/forge.mjs selfcheck
```

浏览器内验证：打开 `dist/ai-forge.html` → 点击「运行全部自检」，即可在浏览器中运行与 Node 端**完全相同**的真实内核（梯度检验 maxRel、FlashAttention≡全注意力、erf 等）。

---

## 四、已实现功能

- **L0 数值内核**：`erf/erfc/lgamma`（NR 连分式）、Kahan 求和、logsumexp、softmax、Welford 在线均值/方差、数值稳定的 `logSigmoid`。
- **L1 自动微分**：tape 反向模式；`matmul/add/tanh/softmaxRows/crossEntropy` 等；中心差分 `gradCheck`（maxRel < 1e-5）。
- **L2 优化器**：AdamW（解耦衰减）、SGD(+Nesterov)、**Muon**（Newton–Schulz 正交化 + RMS 重缩放，1-D 参数回退 AdamW）、L-BFGS（两环递归 + 强 Wolfe 线搜索）。
- **L3 模块**：RMSNorm（‖y‖²/d=1）、RoPE（⟨R(q,m),R(k,n)⟩ 仅依赖 m−n）、SwiGLU、MoE top-k 路由（辅助损失均匀时=1）、确定性初始化。
- **L4 注意力**：FlashAttention（在线 softmax，O(1) 额外显存，数值≡全注意力）、GQA、MLA（低秩 KV 压缩，KV cache 4× 更小）、KV-cache 增量解码 bit-exact。
- **L5 Transformer**：pre-norm、tied head、GQA、可选 MoE；`forward`/`generate`（贪心/采样）。
- **L6 经典 ML**：OLS/Ridge/LASSO/Logistic、SMO-SVM(KKT)、CART/RF/GBDT、k-means++/GMM-EM、KD-tree/KNN/NB、PCA/kernel-PCA。
- **L7 推理**：top-k / top-p / min-p 采样、beam search、投机解码（无损接受率）、困惑度。
- **L8 验证**：52 条可执行不变量，覆盖每层数学声明（正交性、残差⊥X、单调性、互补松弛、Eckart–Young 误差……）。
- **L9/L10**：量化基准 + 零依赖单文件打包（18 模块 → 130KB，浏览器直接跑）。

---

## 五、已知限制

- **Transformer 训练路径**：`TransformerLM.forward` 为数值（非 tape）前向，已验证 flash≡full、确定性、贪心生成；端到端反向传播训练复用 L2 的 autodiff MLP 路径验证（不变量明确标注「full training path proven in L2」）。将 Transformer 接入 autodiff 图是下一阶段路线（见 ROADMAP）。
- **GEMM**：手写 blocked ikj 已做分块，但在小矩阵上受 V8 JIT 影响可能不优于朴素 ijk；生产吞吐建议接入 WASM/WebGPU 后端（零重写可插拔）。
- **单线程**：所有基准为单线程 V8，数字为指示性；复杂度声明才是被不变量校验的部分。
- **MoE 公平性**：Muon vs AdamW 的「谁更快」取决于学习率与问题，本系统只断言 Muon 能收敛到近最优（≤2% 初值），不作脆弱的胜负门禁。

---

## 六、署名

**作者 / 署名：晨星（Chenxing）** · MIT License · 完整代码与文档开源。

> 「零重写、集大成、可验证」—— 不做轮子，只锻造世界最优解的合金。
