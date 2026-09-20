# microjev

[![tests](https://github.com/day253/microjev/actions/workflows/tests.yml/badge.svg)](https://github.com/day253/microjev/actions/workflows/tests.yml)

A dependency-free educational transformer for **typed probabilistic decisions**.

用纯 Python 从零训练一个小 Transformer，一次读取文本后，直接输出多个问题的概率分布和结构化结果。受 Karpathy 的 [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 和 TypeSafe 的 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。

这是独立教学项目，与 TypeSafe 没有隶属关系。没有使用 Jev 权重，也没有实现其未公开的 RLCD 或内部架构。

## 一分钟开始

Python 3.9+，运行不需要第三方依赖、GPU、API key 或下载数据。

```bash
git clone https://github.com/day253/microjev.git
cd microjev
python3 -m microjev demo --steps 300 --output runs/demo.json
python3 -m microjev predict --model runs/demo.json \
  --text '工单：加急，退款，影响严重'
```

第一条命令训练内置的中文合成工单数据，用 validation 模板拟合温度，再评估独立 test 模板并保存模型。日志输出到 stderr，结果 JSON 输出到 stdout。纯 Python 的实际耗时取决于机器和配置。

需要命令行入口时，可在虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
microjev --help
```

## 输出类型

- **Choice**：给固定候选选项分配概率，返回概率最大的选项。
- **Noul**：返回命题为真的概率 `noul`，范围 0–1；不自动转成 bool。
- **Score**：预测有序等级分布，返回等级下标的期望 `sum(i * p[i])`。三个等级的范围为 0–2，可以是小数。

Choice 和 Score 的 `confidence` 使用本项目定义的 **1 − 归一化熵**。它衡量分布集中程度，不是正确率，也不声称与 TypeSafe 的 confidence 公式相同。Noul 没有额外的 confidence 字段。

返回的 `metadata.calibration` 表示是否拟合过温度；`temperature_fitted` 不代表可靠性已得到保证。`metadata.unknown_characters` 提示输入中有多少字符映射到了 UNK。

## 训练自己的任务

1. 定义固定问题和候选项，参考 [examples/support.schema.json](../examples/support.schema.json)。
2. 准备 UTF-8 JSONL，每行一个 `text` 和对应的 `labels`。
3. 分开准备训练、校准和测试集，相关客户、来源或模板不要泄漏到不同集合。
4. 训练后用 validation 集拟合温度，最后只用 test 集评估。

一行训练数据：

```json
{"text":"类型退款，加急，影响严重","labels":{"intent":"refund","urgent":true,"severity":2}}
```

Choice 标签是候选项名称；Score 标签是从 0 开始的整数等级；Noul 标签支持 bool 或 0–1 的软概率标签。每行必须包含 schema 的全部问题。

```bash
python3 -m microjev train \
  --schema examples/support.schema.json \
  --data examples/support.train.jsonl \
  --steps 300 --n-embd 16 --n-head 4 --n-layer 1 --block-size 64 \
  --output runs/support.json

python3 -m microjev calibrate \
  --model runs/support.json \
  --data examples/support.validation.jsonl \
  --output runs/support-calibrated.json

python3 -m microjev evaluate \
  --model runs/support-calibrated.json \
  --data examples/support.test.jsonl

python3 -m microjev predict \
  --model runs/support-calibrated.json \
  --text '工单：加急，退款，影响严重'
```

`evaluate` 按问题报告 NLL、Brier score 和 argmax accuracy；Score 的 accuracy 指等级分类正确率，并非连续评分误差。对于 Noul 软标签，accuracy 比较两个分布的 argmax，NLL/Brier 更适合评价概率质量。NLL/Brier 越低越好。

校准是在固定正温度网格上最小化 validation NLL。不要在测试集上拟合温度，也不要把 demo 的合成数据成绩当作真实工单能力。

## Python API

```python
from microjev import MicroJev
from microjev.demo import SCHEMA, dataset

model = MicroJev.from_rows(SCHEMA, dataset("train"), block_size=64)
model.fit(dataset("train"), steps=300)
model.calibrate(dataset("validation"))
model.save("runs/model.json")

loaded = MicroJev.load("runs/model.json")
decision = loaded.predict("工单：加急，退款，影响严重")
print(decision["answers"]["intent"]["choice"])
print(decision["answers"]["urgent"]["noul"])
print(decision["answers"]["severity"]["score"])
```

Checkpoint 是 JSON，包含字符表、schema、网络配置、全部权重和温度，无 pickle、无远程代码执行。加载时校验版本、权重形状和有限数值。它不保存 Adam 状态；再次调用 `fit` 会启动新的优化器和学习率计划，并重置校准。

## 实现方式

输入字符 → BOS/字符/EOS → 因果 Transformer → EOS 隐藏向量 → 多个固定决策头 → 概率和 Python 字典。

保留 microgpt 的标量自动微分思路：token/position embedding、多头注意力、RMSNorm、ReLU MLP、残差连接。每个决策头用交叉熵训练，共享输入表示。训练使用带梯度裁剪和线性学习率衰减的 Adam；推理直接计算 float，不构建自动微分图。

模型不生成输出 token。输入依然由 Python 循环依次处理，多个输出头也用循环计算；没有 GPU 并行加速，也没有 Jev 的延迟保证。

## 第一版范围

- **固定 schema**。`instructions` 和候选描述是文档元数据，模型不会把它们当作指令读取。更改问题含义、候选数量或候选顺序后，应重新训练；不能在推理时传任意新问题。
- **从零训练**。没有预训练语言知识；内置数据仅演示关键词及模板组合，不能处理真实客服语义、否定或推理。
- **字符级分词**。训练集中没见过的字符映射到 UNK；目前没有 token dropout 专门训练 UNK，所以分布外输入通常不可靠。
- **长度上限**。`block_size` 包括 BOS 和 EOS。默认 64 即最多 62 个输入字符，超长明确报错，不静默截断。
- **类型正确不等于判断正确**。输出由程序按类型构造，错误分类或过度自信仍可能发生。

候选项随请求变化时，后续设计需要编码 `state + question + candidate`，使用共享打分器；这个版本尚未实现。项目另有 GPT-2 124M 的 MLX 后端，见 [主 README](../README.md) 和 [Apple Silicon 说明](apple-silicon.md)。

## 开发与验证

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖共享计算图、深图反传、极端 logits、Transformer 梯度的有限差分校验、最小任务学习、float/自动微分前向一致性、checkpoint 往返、损坏文件拒绝、温度拟合、类型语义、输入验证和 CLI。

源码：`microjev/autograd.py` 为自动微分；`model.py` 为模型、训练与校准；`schema.py` 为输出约束；`cli.py` 为命令行；`demo.py` 为合成数据。

## 参考

- [Karpathy microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)：纯 Python GPT 教学实现。
- [Jev 发布文章](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：结构化决策模型的动机。
- [TypeSafe primitives](https://docs.typesafe.ai/primitives)：Choice、Score、Noul 的接口语义。
- [TypeSafe confidence](https://docs.typesafe.ai/confidence)：概率分布与置信度的区别。

MIT License。第三方项目、名称和链接归各自所有者所有。
