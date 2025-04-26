
English-Vietnamese Translation Fine-Tuning Project
This project fine-tunes a pre-trained machine translation model (Helsinki-NLP/opus-mt-en-vi) on the OPUS-100 English-Vietnamese dataset.

📚 Project Description
We load an English-Vietnamese parallel corpus, preprocess it, and fine-tune a sequence-to-sequence (Seq2Seq) transformer model to improve translation quality.
The evaluation uses BLEU and ROUGE metrics to measure the model's performance.

📦 Dependencies
transformers

datasets

evaluate

numpy

torch

Install them using:

bash
Copy
Edit
pip install transformers datasets evaluate numpy torch
🚀 Steps
Load the Dataset

Load opus100 dataset for English-Vietnamese translation.

Preprocessing

Tokenize input (English) and target (Vietnamese) sentences.

Truncate sequences to maximum lengths.

Prepare Data Collator

Use DataCollatorForSeq2Seq to dynamically pad inputs and labels during batching.

Define Training Arguments

Specify batch size, learning rate, number of epochs, and save strategy.

Metrics for Evaluation

Compute BLEU for precision on n-grams.

Compute ROUGE for overlapping n-grams and sequence similarity.

Train the Model

Fine-tune using Seq2SeqTrainer.

Evaluate

Evaluate the model on the validation set and report BLEU and ROUGE scores.

🛠️ Key Files
Preprocessing Function: Prepares the inputs and labels for training.

Compute Metrics Function: Calculates BLEU and ROUGE after decoding predictions.

Training Script: Defines model training using Seq2SeqTrainer.

📈 Evaluation Metrics
BLEU: Measures the precision of n-gram overlaps between prediction and reference.

ROUGE: Measures the overlap of n-grams and sequences (ROUGE-1, ROUGE-2, ROUGE-L).

📂 Output
Fine-tuned model saved in the marianMT-finetuned-en-vi directory.

Logs and evaluation results printed during and after training.
