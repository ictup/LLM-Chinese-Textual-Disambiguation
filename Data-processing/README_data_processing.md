# 语义去重与数据提取工具包

一个基于 Python 的工具包，用于文本数据集的**精确去重**与**语义去重**（特别适用于中文歧义句数据集），并支持**结构化列提取**以便后续处理。

---

## 功能特点

* **精确去重**
  对句子进行规范化（去除标点、转小写、压缩空白）后进行匹配，快速删除完全重复的句子。

* **语义去重**
  使用 `SentenceTransformer` 模型（默认：`intfloat/multilingual-e5-base`）生成句子嵌入，通过余弦相似度计算检测语义重复。

* **高效计算**
  使用 PyTorch（支持 GPU）加速嵌入生成；采用矩阵运算高效计算相似度；整合 `tqdm` 进度条实时监控进度。

* **详细报告生成**
  输出文件包括：

  * 精确去重结果文件
  * 精确 + 语义去重详细报告（含相似度分数）
  * 带删除标记的最终数据集

* **灵活的数据提取**
  提供独立脚本，利用 `pandas` 从 CSV 或 Excel 文件中提取指定关键列（如歧义句、上下文、消歧选项等）。

---

## 环境要求

* Python 3.8+
* PyTorch
* sentence-transformers
* numpy
* pandas
* tqdm

安装依赖：

```bash
pip install torch sentence-transformers numpy pandas tqdm
```

---

## 使用方法

### 去重处理

```bash
python deduplicate.py
```

* 在 `main()` 部分配置文件路径和参数。
* 输出文件：

  * 去重后的数据集 CSV
  * 重复项报告 CSV
  * 精确去重结果 CSV

### 列提取处理

```bash
python extract_columns.py
```

* 在 `main()` 中修改 `input_file` 和 `output_file` 路径。
* 提取并保存指定列。

---

## 核心参数

| 参数                   | 描述                              |
| -------------------- | ------------------------------- |
| `model_name`         | SentenceTransformer 模型名称（默认多语言） |
| `exact_threshold`    | 精确匹配去重阈值                        |
| `semantic_threshold` | 语义去重余弦相似度阈值                     |
| `ambiguity_column`   | 包含需去重句子的列名                      |

---

## 输出文件说明

* **去重后数据集**：带有“是否删除”标记的最终数据。
* **重复项报告**：列出重复句对及其相似度的详细报告。
* **精确去重结果**：记录规范化后的句子及其原始行号。

---

## 注意事项

* 语义去重阶段可能需要较高 GPU 资源；如无 GPU，可适当调整批量大小。
* 阈值可根据数据集特点进行调整优化。
* 如有需要，可扩展支持并行计算等性能优化。
