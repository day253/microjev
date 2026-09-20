# GPT-2 在真实 SST-5 标注上的 Jev 风格决策训练

使用 [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) 提供的 Stanford Sentiment Treebank 句子级划分。该数据是英文影评的五级情感标签；不是客服工单，也不是任意指令数据。

数据固定为 revision `e51bdcd8cd3a30da231967c1a249ba59361279a3`。下载时记录 SHA-256，原始数据与转换后的数据保存在用户指定目录，不提交到 Git。来源及分割记录见 [manifest](benchmarks/sst5-data-manifest.json)。

## 三种输出如何获得标签

- Choice `sentiment`：原标签 0/1 → negative；2 → neutral；3/4 → positive。
- Noul `positive`：原标签 3/4 为 true，其余为 false。
- Score `rating`：原标签 0–4，依次为 very negative、negative、neutral、positive、very positive。

三种任务都是同一个人工标注的投影，并不是三个独立标注来源。模型仍使用独立输出头，预测时不强制三个头之间的逻辑一致性。输入不会拼接标签或标签描述。[schema](../examples/sst5.schema.json)

## 划分和评估顺序

保留原始训练集 8,544 条和测试集 2,210 条。原始 dev 的 1,101 条用 seed=42 打乱，再分为：

- 550 条 selection：每轮评估平均 NLL，选择最佳检查点。
- 551 条 calibration：训练结束后，仅对选中的检查点拟合 softmax 温度。

测试集只在完成训练、选模型和校准后评价一次。测试准确率分别表示三分类、二分类（概率阈值 0.5）和五分类 argmax 的正确率；Score 返回值仍是 0–4 上的概率期望。

## 运行

```bash
.venv/bin/python -m microjev.sst5 \
  --base-model openai-community/gpt2 \
  --data-dir data/sst5 \
  --output runs/gpt2-sst5 \
  --epochs 3 --batch-size 8 --learning-rate 2e-5 --max-length 128

.venv/bin/microjev-mlx predict --model runs/gpt2-sst5/best \
  --text 'A wonderful film with moving performances and a brilliant story.'
```

加载 GPT-2 预训练权重后全量更新主干和决策头。右侧 padding 不会影响池化位置；选择文本末尾 EOS 的隐藏向量。默认每轮 1,068 步，共 3,204 步。

每轮写入 `checkpoints/epoch-N`，`progress.json` 记录选择集结果；训练结束后将最佳检查点校准并保存在 `best`，完整报告保存在 `training_report.json`。Checkpoint 包含 tokenizer，可离线推理；三个轮次及最终模型会占用约 2 GB 磁盘空间。

本任务验证的是英文电影评论的情感决策能力，不表示复现了 Jev 的通用知识、任意问题接口或 RLCD。不能把这里的训练时间外推成从零预训练 GPT-2 的时间。

## 实测结果

M4 Pro / 48 GB，2026-09-20：总耗时 **522.95 秒（8 分 43 秒）**，训练循环含每轮验证与保存耗时 500.49 秒。三轮选择集平均 NLL 为 0.7748、0.8344、1.3007，选择第一轮；后续训练出现过拟合。

校准后，在 2,210 条独立测试数据上，三分类情感准确率 **75.70%**（NLL 0.6040），是否正面准确率 **86.92%**（NLL 0.3297），五级情感 argmax 准确率 **52.35%**（NLL 1.0902）。[完整报告](benchmarks/sst5-training.json)
