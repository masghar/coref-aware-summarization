from transformers import AutoModel, AutoTokenizer

# Replace this with the exact version required
model = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")

model.save_pretrained("models/spanbert-large-cased")
tokenizer.save_pretrained("models/spanbert-large-cased")
