# 最终模型：GPT-2 Jev 风格动态决策

训练与独立测试于 2026-09-21 完成。推荐动态候选检查点为 `runs/candidate-final`，保留的固定头基线为 `runs/gpt2-sst5/best`。两者均可在本机离线加载。权重保存在本地，不包含在 Git 仓库中。

## 立即运行

连续输入模式，在本项目目录执行：

```bash
HF_HUB_OFFLINE=1 .venv/bin/microjev-candidate \
  --model runs/candidate-final \
  --questions examples/dynamic.schema.json \
  --interactive
```

出现 `state>` 后输入文本并回车，每行独立输出一份 JSON。模型和问题定义只加载一次；空行跳过，输入 `exit` / `quit` 或按 Ctrl-D / Ctrl-C 退出。超长输入会报错并允许继续输入。省略 `--state`、`--state-file` 时也会自动进入此模式。可以输入任意文本，示例问题仍按影评情感解读；当前模型仅验证了英文影评能力。

在本项目目录执行：

```bash
HF_HUB_OFFLINE=1 .venv/bin/microjev-candidate \
  --model runs/candidate-final \
  --questions examples/dynamic.schema.json \
  --state 'A wonderful film with moving performances and a brilliant story.'
```

`examples/dynamic.schema.json` 包含 Choice 情感分类、Noul 正面概率、Noul 非正面概率和 Score 五级评分。你可以更换问题 ID、指令和候选描述；模型会读取问题及候选文本。输入长度默认最多 192 tokens，包含问题、候选和 EOS；超长输入报错。支持最多 1024 个位置，但本次训练文本较短，长文本能力未经验证。

长状态可增加 `--max-length 1024 --prefix-cache`。短输入缓存可能稍慢，默认关闭。当前本地 `.venv` 指向本次训练使用的独立环境；新机器请按 README 创建环境，并复制整个 `runs/candidate-final` 文件夹或按实验日志重新训练。

Python API：

```python
import json
from microjev.candidate import CandidateDecision, predict

model, tokenizer = CandidateDecision.load("runs/candidate-final")
with open("examples/dynamic.schema.json", encoding="utf-8") as f:
    questions = json.load(f)
result = predict(model, tokenizer, "A wonderful film.", questions)
print(json.dumps(result, indent=2))
```

## 模型与数据

- 基座：`openai-community/gpt2`，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，124,440,576 个参数（主干加共享标量头），MLX float32。
- 设备：Apple M4 Pro，48 GB 内存；MLX 0.32.2 / MLX-LM 0.31.3。
- 数据：[SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5)，revision `e51bdcd8cd3a30da231967c1a249ba59361279a3`，8,544 条训练影评。所有训练问题标签由同一五级人工情感标签派生，并非独立任务或新增知识标注。
- 划分：550 条选择集、551 条独立校准集、2,210 条测试集。选择集同时用于检查改写和否定问题，不能将它当成最终独立评测。
- 保留权重链：原始 GPT-2 → 固定头基线 B 的第 1 轮 → 候选 C2 的第 1 轮 → 配对 C3 的第 1 轮 → 补充条件 C4 的第 1 轮。各阶段重复使用相同训练划分，完整记录见[实验日志](iteration-log.md)。
- C4 在最终测试前被选定并冻结。随后只在校准集拟合温度：Choice 1.5、Noul 1.5、Score 2.0。未根据最终测试结果继续调整模型。

## 独立测试结果

每项均评估同一批 2,210 条测试影评。下列任务彼此相关，不应当当成多个独立数据集的证据：

- 标准三分类情感：准确率 **71.54%**，NLL 0.7531，Brier 0.4244。
- 标准正面命题：准确率 **87.19%**，NLL 0.3629，Brier 0.2100。
- 五级评分 argmax：准确率 **51.09%**，NLL 1.3075，Brier 0.6587；返回的 Score 本身是 0–4 的概率期望。
- 正面改写命题：准确率 **86.83%**；负面改写命题：**83.08%**；带中性说明的非正面命题：**86.43%**。
- 无含义选项名称：三选一 **69.14%**、五选一 **50.32%**、二选一 **77.42%**。

[最终校准与测试报告](benchmarks/candidate-final.json)。固定头基线测试三分类为 75.70%、五分类为 52.35%，在这两项固定任务上更强；动态模型支持运行时问题和候选，不是所有指标都优于固定头。[固定头报告](benchmarks/sst5-training.json)

## 输出与推理

Choice、Noul、Score 都由程序按概率构造，不生成再解析 JSON。Noul 是命题为真的概率；Score 是等级分布期望；Choice 返回每个候选的概率。候选换序不会在浮点误差范围外改变其对应概率，问题 ID 不进入模型。

跨问题批量打分已默认启用。可选共同前缀缓存只在单次请求中复用，不跨请求保存状态。十候选、三个问题的同步 GPU 打分实测：短状态 17.07 ms；131-token 状态缓存后 22.58 ms；561-token 状态缓存后 38.89 ms，对同样批处理设置约快 5.1 倍。测试不含 tokenizer/JSON 开销，长文本使用重复的自写句子，仅验证速度和数值等价。[推理基准](benchmarks/candidate-inference.json)

最终保存模型又在 32 条选择集记录上验证了缓存、分批、候选换序和问题 ID 改名：最大概率差为 2.58e-6，低于事先设置的 1e-5 阈值。上述速度基准的三组样例偏差低于 8e-7。

[自写例句的实际输出](benchmarks/final-examples.json)，[离线加载、数值等价与权重校验记录](benchmarks/final-verification.json)。

## 能力边界

这是 GPT-2 的英文影评情感决策模型，没有 Jev 权重，没有复现其私有架构、RLCD 或通用知识能力。支持任意格式合法的动态问题，不代表能正确回答任意问题。中文、其他业务领域和长文档需要新的数据与独立评估。

confidence 使用本项目的 `1 - normalized entropy`，不等于正确概率，也不是 TypeSafe 的原始公式。温度在 SST-5 标准问题上拟合，对其他问题和分布外输入没有一般性校准保证。多个问题独立打分，正反命题的概率不被强制相加为 1。

如需继续训练，应作为新的实验明确评估安排，不再用本报告的测试结果挑选本轮改动。仓库 MIT 许可覆盖原创代码；依赖、原始模型和数据遵循其各自许可。
