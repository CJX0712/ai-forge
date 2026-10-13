# 路线图 · AI Forge / 星锻

> 「零重写、集大成、可验证」的下一步。每条都可独立插拔，不破坏现有不变量。

## 已完成（v1.0.0）

- [x] L0–L1 数值内核 + 自动微分（gradCheck maxRel < 1e-5）
- [x] L2 优化器：AdamW / SGD / **Muon** / L-BFGS
- [x] L3–L5 RMSNorm/RoPE/SwiGLU/MoE · FlashAttention/GQA/MLA · TransformerLM
- [x] L6 经典 ML 全谱系（linear/svm/tree/cluster/instance/pca）
- [x] L7 推理采样（top-k/top-p/min-p/beam/投机解码）
- [x] L8 **52 条可执行不变量**（52/52 绿）
- [x] L9 量化基准（含诚实基线）
- [x] L10 零依赖单文件打包 `dist/ai-forge.html`（130KB，浏览器直跑）
- [x] CLI `forge`（demo/test/bench/selfcheck/html/version）
- [x] 完整文档（README / ARCHITECTURE / BENCHMARKS / ROADMAP）+ MIT

## 近期（v1.1）

- [ ] **Transformer 接入 autodiff 图**：把 `TransformerLM.forward` 改为 tape 式前向，使其可被 `backward` 端到端训练（目前训练路径复用 L2 MLP 验证；不变量已标注）。打通后增加「LM 训练 loss 单调下降」不变量。
- [ ] **WebGPU/WASM GEMM 后端**：在 `linalg.gemm` 上插拔高性能后端，零算法改动；新增「backend 一致性」不变量（WASM vs JS 数值一致）。
- [ ] **量化推理**：INT8/FP8 权重 + 动态定点；新增「量化误差上界」不变量。
- [ ] **更多注意力变体**：滑动窗口、线性注意力（RetNet/Mamba 风格状态空间），复用现有 `attentionFull` 接口。

## 中期（v1.2+）

- [ ] **分布式/Muon 扩展**：把 Muon 的 Newton–Schulz 扩展到分片权重（参考 Moonshot 分布式 Muon）。
- [ ] **训练循环工具箱**：`Trainer`（数据迭代、混合精度、梯度裁剪、checkpoint），全部由不变量守护。
- [ ] **模型动物园**：用现有模块拼出可运行的小模型（微型 GPT / 微型 BERT / 微型 ViT），附复现脚本。
- [ ] **WASM 构建目标**：把整个库编译为 WASM，供非 JS 宿主调用。

## 长期（愿景）

- [ ] 成为「AI 系统教学/研究的事实标准零依赖基线」：每个算法都带可执行不变量与论文链接，新人可在一文件内读懂并验证现代 AI 系统的全部支柱。
- [ ] 持续「集大成」：任何更强的成熟方案（新优化器、新注意力、新归一化）出现即评估并替换自写实现。

## 贡献约定

- 任何新模块**必须**配套至少一条可执行不变量（数学声明验证到机器精度）。
- 任何性能优化**必须**保留与基线对照的基准项，且不破坏既有不变量（CI 门禁 `forge selfcheck`）。
- 署名统一为 **晨星（Chenxing）**，许可证 MIT。

---
署名：晨星（Chenxing）· MIT
