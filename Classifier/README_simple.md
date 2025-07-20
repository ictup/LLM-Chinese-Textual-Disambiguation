# 歧义句分类与模型训练工具包

一个基于 Python 和 Transformers 的中文歧义句检测工具，提供从数据预处理、特征增强到模型训练、评估、预测的完整流程。

---

## 功能亮点

* **数据增强**
  使用 `jieba.posseg` 提取词性信息，丰富输入特征。

* **数据集划分**
  按照类别分布，划分训练、验证、测试集，确保平衡。

* **模型支持**
  基于 `hfl/Chinese-RoBERTa-wwm-ext` 等预训练模型进行二分类任务。

* **训练与早停**
  使用 AdamW 优化器、学习率调度器，带有早停机制防止过拟合。

* **详细评估**
  提供准确率、精确率、召回率、F1 分数，输出混淆矩阵与错误分析。

* **预测与保存**
  可单句预测歧义性，支持模型与分词器保存和加载。

---

## 环境依赖

* Python 3.8+
* PyTorch
* transformers
* scikit-learn
* jieba
* numpy
* pandas
* tqdm

安装依赖：

```bash
pip install torch transformers scikit-learn jieba numpy pandas tqdm
```

---

## 使用方法

### 运行完整训练流程

```bash
python main.py
```

配置参数：

* `data_file`: 数据文件（CSV 格式）
* `model_name`: 预训练模型名称（如 hfl/Chinese-RoBERTa-wwm-ext）
* `output_dir`: 模型保存目录

### 示例调用

在代码中调用：

```python
model, tokenizer, error_analysis = run_training_pipeline(
    data_file="提取后的数据集.csv",
    model_name="hfl/Chinese-RoBERTa-wwm-ext",
    output_dir="./ambiguity_detection_model_FINAL",
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01
)
```

### 单句预测

```python
result, prob, ambig_prob, non_ambig_prob = predict_ambiguity(text, model, tokenizer)
```

---

## 核心参数说明

| 参数              | 描述      |
| --------------- | ------- |
| `num_epochs`    | 训练轮数    |
| `batch_size`    | 批次大小    |
| `learning_rate` | 学习率     |
| `weight_decay`  | 权重衰减    |
| `model_name`    | 预训练模型名称 |

---

## 输出内容

* **模型文件**：保存训练好的模型与分词器。
* **评估报告**：测试集指标与部分错误样本分析。
* **预测接口**：用于新样本的歧义性预测。

---

## 注意事项

* 需 GPU 设备以获得最佳训练性能。
* 数据文件需包含列：`歧义句`、`歧义句及上下文`、`歧义句消岐1`、`歧义句消岐2`。
* 可根据实际需求调整超参数和早停策略。

