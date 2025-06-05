# app.py

import time
import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import allennlp_models.coref
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import os
import difflib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import re

os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATHS = {
    "bart-large-cnn (CNN/DailyMail)": "models/bart-large-cnn",
    "bart-large-xsum (XSum)": "models/bart-large-xsum",
    "bart-large-cnn-samsum (SAMSum)": "models/bart-large-cnn-samsum"
}
SPANBERT_MODEL_PATH = "models/spanbert-coref-model.tar.gz"
LOCAL_SPANBERT_PATH = "models/spanbert-large-cased"

@st.cache_resource(show_spinner=False)
def load_models():
    archive = load_archive(SPANBERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_SPANBERT_PATH, local_files_only=True)
    predictor = Predictor.from_archive(archive, predictor_name="coreference_resolution")
    predictor._model._text_field_embedder._token_embedders["tokens"].tokenizer = tokenizer
    return predictor

@st.cache_resource(show_spinner=False)
def load_bart_model(model_path):
    tokenizer = BartTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return tokenizer, model

def resolve_coref(predictor, text):
    coref_out = predictor.predict(document=text)
    words, clusters = coref_out["document"], coref_out["clusters"]
    resolved = list(words)
    replaced_indices = set()

    for cluster in clusters:
        main_mention = words[cluster[0][0]:cluster[0][1] + 1]
        main_text = " ".join(main_mention)

        for mention in cluster[1:]:
            start, end = mention[0], mention[1] + 1
            mention_tokens = words[start:end]

            if all(word.lower() in {"he", "she", "they", "it", "him", "her", "them", "his", "their", "its"} for word in mention_tokens):
                if any(i in replaced_indices for i in range(start, end)):
                    continue
                resolved[start:end] = [f"<REF>{main_text}</REF>"] + [""] * (end - start - 1)
                replaced_indices.update(range(start, end))

    resolved_text = " ".join(filter(None, resolved))
    resolved_text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', resolved_text)
    resolved_text = resolved_text.replace("<REF>", "").replace("</REF>", "")
    return resolved_text

def summarize(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        num_beams=2,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

def compute_rouge_scores(summary1, summary2):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(summary1, summary2)
    return {k: round(v.fmeasure * 100, 2) for k, v in scores.items()}

def compute_bleu_meteor(reference_text, candidate_text):
    if not reference_text or not candidate_text:
        return 0.0, 0.0
    reference_tokens = [reference_text.split()]
    candidate_tokens = candidate_text.split()
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing) * 100
    meteor = meteor_score([reference_text.split()], candidate_text.split()) * 100
    return round(bleu, 2), round(meteor, 2)

def highlight_diff(text1, text2):
    diff = difflib.ndiff(text1.split(), text2.split())
    result = []
    for word in diff:
        if word.startswith('  '):
            result.append(word[2:])
        elif word.startswith('- '):
            result.append(f"**{word[2:]}**")
        elif word.startswith('+ '):
            result.append(f"__{word[2:]}__")
    return " ".join(result)


# Streamlit UI

st.markdown("""
    <style>
        .navbar {
            position: sticky;
            top: 0;
            z-index: 100;
            background-color: #0e1117;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #222;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .navbar img {
            height: 36px;
            margin-right: 10px;
        }
        .navbar-links a {
            margin-left: 15px;
            color: #f0f0f0;
            text-decoration: none;
            font-weight: bold;
        }
        @media (prefers-color-scheme: light) {
            .navbar {
                background-color: #f5f5f5;
                color: #000;
                border-color: #ccc;
            }
            .navbar-links a {
                color: #111;
            }
        }
    </style>
    <div class="navbar">
        <div style="display: flex; align-items: center;">
            <span><strong>Smart Summarizer</strong></span>
        </div>
        <div class="navbar-links">
            <a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank">Paper</a>
            <a href="https://demo.allennlp.org/coreference-resolution" target="_blank">Presentation</a>
        </div>
    </div>
""", unsafe_allow_html=True)

st.title("Coreference-Aware Summarizer")

model_key = st.selectbox("Select Summarizer Model", list(MODEL_PATHS.keys()))
text_input = st.text_area("Enter Document", height=200)

if st.button("Resolve & Summarize") and text_input.strip():
    with st.spinner("Loading models and processing..."):
        start_time = time.time()

        predictor = load_models()
        tokenizer, model = load_bart_model(MODEL_PATHS[model_key])

        baseline_summary = summarize(text_input, tokenizer, model)
        resolved_text = resolve_coref(predictor, text_input)
        pipeline_summary = summarize(resolved_text, tokenizer, model)

        orig_len = len(text_input.split())
        resolved_len = len(resolved_text.split())
        base_len = len(baseline_summary.split())
        pipe_len = len(pipeline_summary.split())

        compression_base = round(100 * (orig_len - base_len) / orig_len, 1)
        compression_pipe = round(100 * (orig_len - pipe_len) / orig_len, 1)

        elapsed_time = round(time.time() - start_time, 2)
        rouge_scores = compute_rouge_scores(baseline_summary, pipeline_summary)
        bleu, meteor = compute_bleu_meteor(baseline_summary, pipeline_summary)

        st.subheader("Resolved Text")
        st.write(resolved_text)

        st.subheader("Baseline Summary (BART Only)")
        st.write(baseline_summary)

        st.subheader("Pipeline Summary (SpanBERT → BART)")
        st.write(pipeline_summary)

        st.markdown(f"**⏱️ Time:** {elapsed_time}s")
        st.markdown(f"**📄 Word Counts:** Original: {orig_len}, Resolved: {resolved_len}, Baseline: {base_len}, Pipeline: {pipe_len}")
        st.markdown(f"**✂️ Compression:** Baseline: -{compression_base}%, Pipeline: -{compression_pipe}%")

        st.markdown("**📐 ROUGE Similarity (Pipeline vs Baseline):**")
        for k, v in rouge_scores.items():
            st.markdown(f"- **{k.upper()}**: {v}%")

        st.markdown(f"**🔹 BLEU Similarity:** {bleu}%")
        st.markdown(f"**🔹 METEOR Similarity:** {meteor}%")

        st.subheader("Summary Differences (Baseline vs Pipeline)")
        st.markdown(highlight_diff(baseline_summary, pipeline_summary))

        fig, ax = plt.subplots()
        labels = ['Original', 'Resolved', 'Baseline', 'Pipeline']
        values = [orig_len, resolved_len, base_len, pipe_len]
        ax.bar(labels, values, color=["gray", "blue", "orange", "green"])
        ax.set_ylabel("Word Count")
        ax.set_title("📊 Compression Comparison")
        st.pyplot(fig)

# Sticky footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            padding: 10px;
            background-color: rgba(0,0,0,0.7);
            color: #f1f1f1;
            text-align: center;
            font-size: 14px;
        }
        @media (prefers-color-scheme: light) {
            .footer {
                background-color: #ffffff;
                color: #333;
                border-top: 1px solid #ccc;
            }
        }
    </style>
    <div class="footer">
        © 2025 Muhammad Asghar | Streamlit | BART | SpanBERT | HuggingFace | AllenNLP
    </div>
""", unsafe_allow_html=True)
