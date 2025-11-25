English to Urdu Neural Machine Translation (NMT)
📌 Project Overview
This project implements a Neural Machine Translation pipeline to translate English text into Urdu using the MarianMT architecture. The system handles the end-to-end workflow, including downloading datasets from Kaggle, cleaning and preprocessing text, and performing inference using a fine-tuned Transformer model.

The core of the translation logic relies on the Encoder-Decoder architecture standard in NMT tasks.

<img width="3840" height="2160" alt="image" src="https://github.com/user-attachments/assets/9d4e8b3e-c2b2-47fb-b6a1-85b8cd821e4d" />


🚀 Features
Automated Data Ingestion: directly fetches the Parallel Corpus for English-Urdu from Kaggle.

Robust Preprocessing: Cleans and normalizes mixed-format text files (TXT/CSV).

Data Splitting: Automatically generates Train, Validation, and Test sets (80/10/10 split).

Inference Pipeline: Loads a saved model for translating new English sentences.

Evaluation: Computes BLEU scores to measure translation quality against reference text.

🛠️ Prerequisites
The project is designed to run in Google Colab but can be adapted for local environments. You will need:

Google Drive (for persistent storage).

Kaggle API Key (kaggle.json) for dataset download.

GPU Runtime (Recommended for inference).

Dependencies
Bash

pip install transformers datasets sentencepiece sacrebleu evaluate accelerate kaggle
📂 Project Structure
The script creates the following directory structure in your Google Drive:

Plaintext

/content/drive/MyDrive/mt_en_ur/
├── parallel_master.csv        # Combined and cleaned dataset
├── train.en / train.ur        # Training pairs
├── valid.en / valid.ur        # Validation pairs
├── test.en / test.ur          # Testing pairs
└── opus_en_ur_minimal_final/  # (Directory containing your fine-tuned model)
⚙️ Usage Guide
1. Data Preparation
The script automatically detects the file format (CSV, TSV, or split TXT files) from the Kaggle download and normalizes them. It applies the following cleaning steps:

Lowercasing.

Whitespace removal.

Regex-based noise reduction.

Python

# Example Cleaning Logic
def clean_text_en(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s
2. Loading the Model
The system uses MarianMTModel, which is highly efficient for translation tasks. Ensure your model path matches the variable in the script:

Python

MODEL_DIR = "/content/drive/MyDrive/mt_en_ur/opus_en_ur_minimal_final"
model = MarianMTModel.from_pretrained(MODEL_DIR)
3. Running Inference
To translate a sentence, the script tokenizes the input, generates the translation tensor, and decodes it back to text.

Python

text = "Where are you going?"
# Tokenize and Generate
batch = tokenizer([text], return_tensors="pt", padding=True).to(device)
generated_ids = model.generate(**batch)
# Decode
translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# Output: "تم کہاں جا رہے ہو؟" (Tum kahan ja rahe ho?)
📊 Evaluation
The model performance is evaluated using the BLEU (Bilingual Evaluation Understudy) score, which compares the n-grams of the candidate translation to the reference translation.

Metric: SacreBLEU

Test Set: test.en vs test.ur

🤝 Acknowledgments
Dataset: Parallel Corpus for English-Urdu Language

Library: Hugging Face Transformers
