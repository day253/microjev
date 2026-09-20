# 在 Apple Silicon 上训练 GPT-2

可以训练，包括从随机权重开始训练 GPT-2 架构，以及加载预训练权重微调。能启动训练、能达到指定验证损失、能复现高速训练纪录，是不同的目标。

## 两条可行路线

**PyTorch + MPS**：Apple 官方支持通过 Metal GPU 后端加速 PyTorch 训练。适合迁移已有 PyTorch 模型。先用 `torch.backends.mps.is_available()` 检查环境，把模型和张量放到 `mps`，再确认所用算子和优化器的兼容性。[Apple 官方文档](https://developer.apple.com/metal/pytorch/)

**MLX**：Apple 的 MLX examples 提供 Transformer language model 从头训练示例，适合围绕 Apple Silicon 编写训练流程。该示例不等于完整 GPT-2 复现；需要另外核对 tokenizer、层归一化、位置编码、权重绑定等配置。[官方示例](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm)

[nanoGPT](https://github.com/karpathy/nanoGPT) 的 README 记录了 `--device=mps`，也包含 GPT-2 模型定义和预训练权重加载。但仓库已声明 deprecated；参考实现可用，实际运行时应验证当前依赖兼容性，不把旧版本速度数字当作新机器上的预测。

## 大家在优化的速度赛道

[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) 很可能就是所说的项目。其主赛道是在 **8 张 NVIDIA H100** 上达到 **FineWeb 验证损失 3.28**，目标来自 GPT-2 复现基线。允许的模型和优化改动意味着它不只是原版 GPT-2 的内核加速。

CUDA/Triton 内核以及多卡通信优化不能仅靠把 device 改成 `mps` 就迁移。可以借鉴优化器、数据调度和训练方法，再在 MLX/MPS 下分别测量训练稳定性、吞吐和验证损失。

## 如何估算本地训练成本

先跑几十步预热，再在真实上下文长度和 batch size 下测量稳定的 tokens/s、峰值内存和验证损失。计时要等待 GPU 工作完成（如 MPS synchronize 或 MLX eval），避免只计异步提交的耗时。

粗略训练时间约为 `总训练 token 数 / 实测 tokens per second`，另计数据准备、评估和保存。仅按参数量或内存容量不能推导速度；在小样本上跑通也不等于复现完整预训练质量。

## microjev 已实现的路线

本项目已实现 MLX 后端，加载 GPT-2 124M 预训练权重后接入固定决策头，支持冻结主干或全量微调。M4 Pro、48 GB 上的短基准、实际微调耗时和测试结果见 [README](../README.md)。这是决策微调流程，没有从零复现 GPT-2 预训练质量或 modded-nanogpt 纪录。

纯 Python 标量自动微分版本保留作教学用途，不适合 GPT-2 规模。

资料核对日期：2026-09-20。
