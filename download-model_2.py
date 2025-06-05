from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")

# Save locally
model.save_pretrained("spanbert-local")
tokenizer.save_pretrained("spanbert-local")
