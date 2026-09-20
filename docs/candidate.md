# 读取运行时问题的共享候选打分器

`CandidateDecision` 将 `state + question.instructions + candidate description` 编码为 GPT-2 输入，用同一个线性层输出每个候选的标量分数。对同一问题的候选分数做 softmax，再按 Choice、Noul、Score 的类型构造结果。模型不生成 JSON 文本。

问题 ID 不进入模型；没有每个问题、每个候选或每种候选数量的专属分类参数。Choice 允许运行时传入 2–255 个候选；Score 支持 2–10 个有序描述；Noul 使用 true/false 两个候选。候选描述会影响分数。问题 ID 改名不会改变结果；候选换序只改变输出排列，在浮点误差范围内保留相同概率。Score 的索引仅决定返回的期望刻度，模型读取等级描述。

这个实现比固定分类头更接近 Jev 的接口。它重复编码每个候选的状态，没有复现 Jev 的共享上下文采样器、RLCD 或未公开架构；速度成本会随候选数量增加。编码了问题也不意味着已经学会任意指令。

## 本机训练

```bash
.venv/bin/python -m microjev.candidate_sst5 \
  --base-model openai-community/gpt2 --data-dir data/sst5 \
  --output runs/candidate-pilot-v1 \
  --train-limit 1024 --selection-limit 550 \
  --epochs 2 --batch-size 4 --learning-rate 1e-5 --max-length 192
```

从原始 GPT-2 预训练权重开始，训练 1,024 条 SST-5 影评对应的 3,072 个问题。`batch-size` 指问题数量；每个问题会展开为多个候选。`--train-limit 0` 使用全部 8,544 条训练数据。候选、状态和问题总长度超过 `max-length` 会报错，不静默丢弃信息。

训练数据对问题做确定性变化：命题分别询问正面、负面、中性、非负面；Score 有时反转等级；Choice 有不同措辞。标签同步变换。它们仍来自同一个人工情感标签，没有增加独立知识来源。

选择集固定为 550 条，每轮按标准三种问题的平均 NLL 选择检查点。额外记录选择集上未训练过的命题措辞和否定问题的准确率、NLL、Brier，检查是否理解问题变化。所有实验使用相同 selection/calibration 划分，不随模型随机种子改变。默认不读取测试标签用于评估，也不拟合温度。

`--source-kind fixed --base-model runs/gpt2-sst5/best` 可从固定头基线的主干开始；`--source-kind candidate` 可从已保存候选模型继续训练。后者会重新初始化优化器，不能算精确恢复。对这类实验必须记录主干此前见过的数据，不能声称只训练了新增子集。

## 动态问题与离线推理

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m microjev.candidate_cli \
  --model runs/candidate-pilot-v1/best \
  --questions examples/sst5.schema.json \
  --state 'A wonderful film with moving performances and a brilliant story.'
```

`--questions` 接收运行时 JSON schema，可更换问题、候选名称和描述。也可使用 `--state-file` 输入 UTF-8 文件。安装后等价命令为 `microjev-candidate`。自定义问题没有性能保证，当前数据只支持英文电影情感方向的评估。

同一 Choice 的候选分数由同一模型独立给出，因此换序不需要额外训练。但新增候选会改变 softmax 分母和原候选的概率；不能将一个候选集合的概率直接解释为另一个候选集合上的概率。多问题输出也不强制逻辑一致。

## 最终校准与测试

全部实验用选择集比较完成后，对选定检查点执行一次：

```bash
.venv/bin/python -m microjev.candidate_finalize \
  --model runs/selected-experiment/best \
  --output runs/candidate-final --data-dir data/sst5
```

该命令不重新训练。它用独立的 551 条校准数据拟合每种输出类型的温度，再评估 2,210 条测试数据，并保存可离线加载的最终权重和 `final_report.json`。温度只在 SST-5 标准问题上验证；元数据明确记录适用范围。不要基于最终测试分数再挑改动。

## 测试覆盖

测试验证：混合候选数量的 loss 与手工 NLL 一致、候选换序及问题 ID 改名、Score 刻度反转、padding 和分批不变性、指令确实进入输入、训练与校准、离线 checkpoint 往返、超长输入拒绝及标签变换正确性。测试保证接口与数值实现；训练后的泛化能力由独立数据评估。

## 首次实验结果

1,024 条影评的 C1 实验共耗时 377 秒。最佳轮次在 550 条选择集上的三分类准确率为 37.09%、正面命题 58.91%、五分类 22.73%，平均 NLL 1.1713；明显弱于固定头基线。改写与否定问题的结果也未证明可靠的指令理解。该 checkpoint 仅用于保留实验，不能作为最终能力成果。十候选、三问题的短输入推理，预热后五次平均 23.18 ms。[完整报告](benchmarks/candidate-pilot-v1.json)

随后做了优化诊断：从固定头基线主干开始，16 条训练影评、48 个问题变体、120 次更新，在约 36 秒内达到训练集三个类型全部 100% 准确率。这个刻意过拟合检查表明优化链路可学习，不能当成泛化结果。[诊断报告](benchmarks/candidate-overfit-check.json)。正在用同一来源主干、全量 8,544 条训练数据进行 C2 实验。

## C2 结果与下一轮改进

全量训练 C2 用时 41 分 42 秒，选择第一轮。选择集准确率：三分类 71.27%、正面命题 83.64%、五分类 48.73%。平均 NLL 为 1.2684；第二轮恶化至 1.9515，模型过度自信。更关键的是，否定命题探针准确率仅 24.36%，说明会做情感分类不等于理解动态问题。[C2 完整报告](benchmarks/candidate-full-v2.json)

针对这些已观察到的错误，增加两个可选训练设置：

- `--augmentation paired`：同一条影评同时提供命题及其逻辑补集，覆盖正面、负面、中性及多种措辞；每条影评共四个问题。原有 `legacy` 仍为默认值，可复现 C1/C2。
- `--label-smoothing 0.1`：仅在训练中把 10% 目标质量分配到均匀分布，抑制概率饱和。原始标签和评估标签不变。

`--selection-objective instruction-balanced` 用标准问题和改写/否定探针的平均 NLL 共同选择模型。探针反复用于开发选择，不能再当成独立的最终能力评测；它们的精确问句仍未进入训练模板。默认 `canonical` 保持原有选择方式。新配置需要通过验证结果证明有效，不能提前声称已经改善。

```bash
.venv/bin/python -m microjev.candidate_sst5 \
  --base-model runs/candidate-full-v2/best --source-kind candidate \
  --data-dir data/sst5 --output runs/candidate-paired-v3 \
  --train-limit 0 --epochs 2 --batch-size 8 --learning-rate 5e-6 \
  --augmentation paired --label-smoothing 0.1 \
  --selection-objective instruction-balanced
```

## C3：配对训练结果

C3 在 C2 保留主干上继续训练，共 49 分 08 秒，按组合选择指标保留第一轮。标准问题平均 NLL 从 1.2684 降到 0.9348；标准三分类、正面命题、五分类准确率分别为 70.18%、82.91%、50.18%。标准 NLL 仍弱于固定头基线的 0.7748。

改写正面、改写负面、否定正面的探针准确率分别为 **82.91%、85.64%、64.18%**。对应 NLL 为 0.4287、0.4360、0.7097，说明配对训练方案有改进；但否定能力仍有限，也未验证其他领域。第二轮组合 NLL 变差，因此保留第一轮。[C3 完整报告](benchmarks/candidate-paired-v3.json)

## 否定问题的措辞诊断

C3 在负面影评上，对熟悉的“不是正面”问法正确率为 95.43%，对简单的新问法为 91.78%；加上探针的“中性算真”补充说明后，正确率降到 51.14%。仅把 NOT 改成小写也没有解决问题。这是条件组合和措辞敏感的问题。[选择集诊断](benchmarks/candidate-v3-negation-diagnostic.json)

新增 `--augmentation boundary` 在各类命题及补命题上加入与标签一致的类别说明，覆盖正面、负面、中性，说明放在问题前或后，并变化 NOT 大小写。保留一部分短问题，开发探针的精确句子仍不进入训练。该设置用于下一轮实验，效果需要实测。
