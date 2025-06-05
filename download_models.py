import os
from transformers import BartForConditionalGeneration, BartTokenizer
from allennlp.models.archival import load_archive

MODEL_PATHS = {
    "bart-large-cnn (CNN/DailyMail)": "models/bart-large-cnn",
    "bart-large-xsum (XSum)": "models/bart-large-xsum",
    "bart-large-cnn-samsum (SAMSum)": "models/bart-large-cnn-samsum"
}

SPANBERT_MODEL_URL = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
SPANBERT_MODEL_PATH = "models/spanbert-coref-model.tar.gz"

os.makedirs("models", exist_ok=True)

# Function to download BART models
def download_bart_model(model_name, save_path):
    print(f"Downloading {model_name}...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Saved {model_name} to {save_path}")

# Download BART models
download_bart_model("facebook/bart-large-cnn", MODEL_PATHS["bart-large-cnn (CNN/DailyMail)"])
download_bart_model("facebook/bart-large-xsum", MODEL_PATHS["bart-large-xsum (XSum)"])
download_bart_model("philschmid/bart-large-cnn-samsum", MODEL_PATHS["bart-large-cnn-samsum (SAMSum)"])

# Download SpanBERT coreference model
def download_spanbert_model(url, output_path):
    import requests
    if not os.path.exists(output_path):
        print("Downloading SpanBERT coreference model...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved SpanBERT model to {output_path}")
    else:
        print("SpanBERT model already exists.")

download_spanbert_model(SPANBERT_MODEL_URL, SPANBERT_MODEL_PATH)
