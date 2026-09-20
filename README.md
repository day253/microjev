# microjev

[![tests](https://github.com/day253/microjev/actions/workflows/tests.yml/badge.svg)](https://github.com/day253/microjev/actions/workflows/tests.yml)

**GPT-2 124M + Jev-style typed decisions, trained locally on Apple Silicon.**

加载 GPT-2 预训练权重，用 MLX 训练结构化决策头，直接输出 **Choice / Noul / Score**。提供全量微调、仅训练决策头、模型保存与离线加载、概率校准和本机速度基准。另附一个零依赖的纯 Python 教学实现。

新增[动态候选打分器](docs/candidate.md)：运行时输入状态、问题和候选描述，由同一个 GPT-2 标量打分头输出分布。已通过候选换序、问题 ID 改名、混合候选数量等本机测试。C4 在选择集上的改写正面/负面问题准确率为 83.27% / 84.36%，否定命题提升到 84.00%；标准问题平均 NLL 为 0.9377，仍弱于固定头基线。接口测试通过不代表通用指令能力。[实验报告](docs/benchmarks/candidate-boundary-v4.json)

这是独立项目，受 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 和 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。没有使用 Jev 权重，不是 Jev 内部架构或 RLCD 的复现。

## Apple Silicon 快速开始

需要 Apple Silicon Mac、支持当前 MLX 的 macOS、Python 3.10+；本项目在 M4 Pro / macOS 26.6.2 / Python 3.13 上验证。使用 [uv](https://docs.astral.sh/uv/) 创建独立环境：

```bash
git clone https://github.com/day253/microjev.git
cd microjev
uv venv --python 3.13
uv pip install -e '.[mlx]'

# 首次下载 GPT-2 权重；训练 288 条合成工单，验证并保存
.venv/bin/microjev-mlx demo --mode full --epochs 3 --output runs/gpt2-demo

# 模型和 tokenizer 已保存在本地，可离线使用
.venv/bin/microjev-mlx predict --model runs/gpt2-demo \
  --text 'I want my payment refunded. Please handle this urgently. All work is blocked and there is no workaround.'
```

不用 uv 时，可用 Python 3.10+ 的 `venv` 和 `python -m pip install -e '.[mlx]'`。GPT-2 权重来自 [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)，保留其许可。本次训练使用 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` 的本地快照。权重不包含在 Git 仓库中。

想快速看算法、无需 GPU 或第三方依赖：

```bash
python3 -m microjev demo --steps 300 --output runs/tiny.json
```

详见 [纯 Python 教学版](docs/pure-python.md)。它不是 GPT-2 124M。

## Jev 风格输出

- **Choice**：返回候选项、每个候选项的概率和 confidence。
- **Noul**：返回“命题为真”的概率 `noul`，0–1，不自动转成 bool。
- **Score**：返回有序等级分布及其期望 `sum(i * p[i])`。三个等级的范围为 0–2。

一次编码输入，所有决策头读同一个隐藏向量；不需要生成或解析文本 JSON。返回的 Python 字典由程序按类型构造。

以上是固定头后端 `microjev-mlx` 的路径。新后端 `microjev-candidate` 分别编码状态、问题与每个候选的组合，支持动态问题与候选。默认跨问题批量计算，可用 `--prefix-cache` 复用请求内共同前缀；实测长输入比相同批处理设置快约 5.1 倍，概率偏差低于 8e-7。详见[架构、限制和命令](docs/candidate.md)。

Choice 和 Score 的 confidence 使用本项目定义的 `1 - normalized_entropy`。它不是正确率，也不声称与 TypeSafe 的 confidence 算法相同。校准使用 validation 集拟合每个问题的 softmax 温度，不能保证分布外概率可靠。

## 训练自己的数据

固定问题定义见 [GPT-2 示例 schema](examples/gpt2-support.schema.json)。UTF-8 JSONL 的每行包含文本和所有问题的标签：

```json
{"text":"Please refund my payment. No rush. The impact is minor.","labels":{"intent":"refund","urgent":false,"severity":0}}
```

Choice 标签是候选项名称；Score 标签是从 0 开始的整数等级；Noul 支持 bool 或 0–1 的软概率标签。

```bash
.venv/bin/microjev-mlx train \
  --schema examples/gpt2-support.schema.json \
  --data examples/gpt2-support.train.jsonl \
  --validation examples/gpt2-support.validation.jsonl \
  --mode full --epochs 3 --batch-size 4 --max-length 128 \
  --output runs/support

.venv/bin/microjev-mlx evaluate --model runs/support \
  --data examples/gpt2-support.test.jsonl
```

`--mode full` 默认以 2e-5 学习率更新所有参数。`--mode heads` 冻结 GPT-2，只训练决策头，默认学习率 1e-3。可用 `--learning-rate` 覆盖。日志走 stderr，结果 JSON 走 stdout。

独立校准可用 `microjev-mlx calibrate --model ... --data ... --output ...`。不要用测试集拟合温度或选超参数；真实数据应按客户、来源或模板分组切分。

保存目录包含完整 `model.safetensors`、tokenizer、`microjev.json` 和训练报告。重新加载无需连接网络，不使用 pickle 或远程自定义代码。当前不保存优化器状态，不支持中断后的精确续训。

## 时间预估与实测

需要真实标注数据训练时，使用 [SST-5 训练说明](docs/sst5.md)：固定版本下载 Stanford 影评情感数据，分开做训练、检查点选择、温度校准和最终测试。

```bash
.venv/bin/python -m microjev.sst5 --epochs 3 --batch-size 8 --output runs/gpt2-sst5
```

它按每轮选择集平均 NLL 保留最佳模型，每轮保存一次完整检查点。最终离线推理使用 `--model runs/gpt2-sst5/best`。

2026-09-20，在 **M4 Pro、48 GB 内存**上，以 MLX 0.32.2、float32、batch=4、序列长度=128 测试 GPT-2 主干加三个决策头，共 **124,445,960 参数**。预热 3 步，测量 10 步，每步同步 GPU 参数与优化器更新：

- 全量微调平均 **0.139 秒/步**，约 **3,686 padded tokens/s**；峰值活跃内存约 **2.87 GiB**。
- 仅训练决策头平均 **0.023 秒/步**，共 6,152 个可训练参数；峰值活跃内存约 **0.96 GiB**。
- 全量微调 1,000 条 × 3 epoch，纯训练外推约 104 秒，实际安排 **2–4 分钟**；10,000 条 × 3 epoch，纯训练外推约 17.4 分钟，实际安排 **20–30 分钟**。首次下载和环境安装另计。

以上是短基准外推，不是持续满载测试。文本长度、padding、后台负载和热降频都会影响结果；这是决策训练吞吐，不是 50,257 类 next-token 预训练吞吐。随机权重基准也不能说明学习质量。[原始测量记录](docs/benchmarks/m4-pro-48gb.json)

复测：

```bash
.venv/bin/python -m microjev.benchmark --mode full \
  --batch-size 4 --sequence-length 128 --warmup 3 --steps 10
```

## 演示训练结果与限制

真实 SST-5 基线已训练完成：8,544 条训练数据、3 epoch，总耗时 **8 分 43 秒**。按独立选择集保留第一轮，另用 551 条数据校准；2,210 条测试数据上三分类情感 **75.70%**、是否正面 **86.92%**、五级情感 **52.35%**。这些都是同一情感标签的不同投影。[完整报告](docs/benchmarks/sst5-training.json)

已实际加载 GPT-2 预训练权重并全量微调 **288 条英文合成工单、3 epoch、216 步**，训练耗时 **20.6 秒**。这里按 batch 内最长文本动态 padding，文本比 128 tokens 短，所以快于上面的定长外推。

18 条未见过的测试改写中：意图分类 **17/18**、紧急度 **9/18**、影响等级 **13/18**。训练误差很低，但紧急度泛化不足，验证集上拟合温度也没有解决。这个结果只证明流程可运行，不是业务可靠性证明。[完整训练报告](docs/benchmarks/gpt2-demo.json)

- **固定头的 schema**：`microjev-mlx` 的问题和候选项由训练确定，instructions 与候选描述只是元数据。`microjev-candidate` 会读取运行时指令和候选，但尚不能据此声称具备通用指令能力。
- **语言与数据**：默认 GPT-2 主要面向英文；示例仅为合成数据。中文和真实业务需要合适的基座、标注数据及独立评估。
- **长度**：默认最多 128 个 GPT-2 BPE tokens，包含末尾 EOS；超长输入报错，不静默截断。模型最多支持 1024 个位置。
- **训练实现**：使用 MLX-LM 的 GPT-2 主干（LayerNorm、GELU、绝对位置编码）。该实现没有原始 GPT-2 训练时的 dropout；这里是预训练权重的决策微调，不声称逐项复现原始预训练配方。
- **类型保证**：类型正确不代表判断正确，高 confidence 也可能答错。

关于 Mac 训练 GPT-2 和 modded-nanogpt 速度赛道，见 [Apple Silicon 说明](docs/apple-silicon.md)。

## 开发与测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖：标量自动微分、Transformer 梯度校验、真实训练收敛、float 前向一致性、checkpoint 往返、温度拟合、输出类型、CLI、MLX padding 不变性、冻结主干和全量更新、离线保存加载、SST-5 标签映射。

GitHub CI 在 Python 3.9 / 3.12 / 3.14 上运行纯 Python 测试；未安装 MLX 时跳过 GPU 后端测试。GPU 后端测试在 Apple Silicon 本地运行。

主要文件：`mlx_backend.py` 为 GPT-2 决策模型；`mlx_cli.py` 为训练与推理；`benchmark.py` 为速度基准；`model.py` 和 `autograd.py` 为纯 Python 教学版；`schema.py` 为共享输出约束。

## 参考

- [Karpathy microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Apple MLX-LM GPT-2 实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gpt2.py)
- [TypeSafe primitives](https://docs.typesafe.ai/primitives) 与 [confidence](https://docs.typesafe.ai/confidence)

MIT License。本仓库原创代码适用该许可；MLX、GPT-2 等依赖和权重适用各自许可。
