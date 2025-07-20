# Semantic Deduplication and Data Extraction Toolkit

A Python-based toolkit for **exact deduplication** and **semantic deduplication** of text datasets (especially suitable for Chinese ambiguous sentence datasets), with support for **structured column extraction** for further processing.

---

## Features

* **Exact Deduplication**
  Matches and quickly removes fully duplicated sentences after normalization (removing punctuation, converting to lowercase, compressing whitespace).

* **Semantic Deduplication**
  Uses the `SentenceTransformer` model (default: `intfloat/multilingual-e5-base`) to generate sentence embeddings, and detects semantically duplicated sentences via cosine similarity calculation.

* **Efficient Computation**
  Utilizes PyTorch (GPU supported) for fast embedding generation, efficient similarity calculation with matrix operations, and integrates `tqdm` progress bars for real-time progress monitoring.

* **Detailed Report Generation**
  Output files include:

  * Exact deduplication result file
  * Detailed report for exact + semantic deduplication (including similarity scores)
  * Final dataset with deletion flags

* **Flexible Data Extraction**
  Provides standalone scripts using `pandas` to extract specified key columns (such as ambiguous sentences, context, disambiguation options, etc.) from CSV or Excel files.

---

## Requirements

* Python 3.8+
* PyTorch
* sentence-transformers
* numpy
* pandas
* tqdm

Install dependencies:

```bash
pip install torch sentence-transformers numpy pandas tqdm
```

---

## Usage

### Deduplication

```bash
python deduplicate.py
```

* Configure file paths and parameters in the `main()` section.
* Output files:

  * Deduplicated dataset CSV
  * Duplicates report CSV
  * Exact deduplication results CSV

### Column Extraction

```bash
python extract_columns.py
```

* Modify `input_file` and `output_file` paths in `main()`.
* Extract and save specified columns.

---

## Core Parameters

| Parameter            | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `model_name`         | SentenceTransformer model name (default multilingual)  |
| `exact_threshold`    | Threshold for exact deduplication                      |
| `semantic_threshold` | Cosine similarity threshold for semantic deduplication |
| `ambiguity_column`   | Column name containing sentences for deduplication     |

---

## Output Files

* **Deduplicated Dataset**: Final data with “to be deleted” flags.
* **Duplicates Report**: Detailed report listing duplicate sentence pairs and their similarity scores.
* **Exact Deduplication Results**: Records of normalized sentences and their original row numbers.

---

## Notes

* Semantic deduplication may require significant GPU resources; if no GPU is available, adjust batch size as needed.
* Thresholds can be adjusted and optimized according to dataset characteristics.
* If necessary, the toolkit can be extended to support further performance optimizations such as parallel computation.
