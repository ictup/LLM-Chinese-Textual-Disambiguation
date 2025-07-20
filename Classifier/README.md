# Ambiguity Sentence Classification and Model Training Toolkit

A Python-based ambiguity sentence detection tool using Transformers, tailored for Chinese text, covering the entire process from data preprocessing and feature enrichment to model training, evaluation, and prediction.

---

## Key Features

* **Data Augmentation**
  Utilizes `jieba.posseg` for part-of-speech tagging to enrich input features.

* **Dataset Splitting**
  Splits data into training, validation, and test sets based on category distribution to ensure balance.

* **Model Support**
  Supports binary classification tasks using pre-trained models like `hfl/Chinese-RoBERTa-wwm-ext`.

* **Training with Early Stopping**
  Uses AdamW optimizer, learning rate schedulers, and early stopping mechanisms to prevent overfitting.

* **Detailed Evaluation**
  Provides metrics such as accuracy, precision, recall, F1 score, outputs confusion matrices, and error analysis.

* **Prediction and Saving**
  Supports single-sentence prediction of ambiguity, with model and tokenizer saving and loading capabilities.

---

## Environment Dependencies

* Python 3.8+
* PyTorch
* transformers
* scikit-learn
* jieba
* numpy
* pandas
* tqdm

Install dependencies:

```bash
pip install torch transformers scikit-learn jieba numpy pandas tqdm
```

---

## Usage

### Running the Full Training Process

```bash
python main.py
```

Configuration Parameters:

* `data_file`: CSV file containing data
* `model_name`: Pre-trained model name (e.g., hfl/Chinese-RoBERTa-wwm-ext)
* `output_dir`: Directory to save trained models

### Example Call in Code

```python
model, tokenizer, error_analysis = run_training_pipeline(
    data_file="processed_dataset.csv",
    model_name="hfl/Chinese-RoBERTa-wwm-ext",
    output_dir="./ambiguity_detection_model_FINAL",
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01
)
```

### Single Sentence Prediction

```python
result, prob, ambig_prob, non_ambig_prob = predict_ambiguity(text, model, tokenizer)
```

---

## Core Parameters

| Parameter       | Description                   |
| --------------- | ----------------------------- |
| `num_epochs`    | Number of epochs for training |
| `batch_size`    | Batch size for training       |
| `learning_rate` | Learning rate for optimizer   |
| `weight_decay`  | Weight decay for optimizer    |
| `model_name`    | Name of pre-trained model     |

---

## Output

* **Model Files**: Saved trained models and tokenizers.
* **Evaluation Report**: Metrics on test set and analysis of error samples.
* **Prediction Interface**: Interface for predicting ambiguity in new samples.

---

## Notes

* GPU acceleration is recommended for optimal training performance.
* Data file should include columns: `ambiguous_sentence`, `ambiguous_sentence_with_context`, `ambiguous_option1`, `ambiguous_option2`.
* Adjust hyperparameters and early stopping strategies based on specific requirements.
