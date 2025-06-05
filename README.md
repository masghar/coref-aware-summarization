**Coreference-Aware Abstractive Summarization**
**Enhancing Coherence in Abstractive Summarization via Anaphora Resolution and BART**

This repository implements a summarization pipeline that improves coherence and referential clarity by resolving coreference chains before summary generation. We combine a SpanBERT-based coreference resolver with BART-based abstractive summarization models, demonstrating improvements in both automatic and human evaluations.

**Features**
Coreference resolution using SpanBERT (via AllenNLP)

Abstractive summarization using BART (CNN/DailyMail, XSum, SAMSum)

Tkinter-based GUI for demonstration and comparison

Offline support for all models (no internet access required)

Evaluation module for ROUGE, BERTScore, and pronoun accuracy

Human evaluation interface for Coherence, Fluency, Factuality, and Clarity

**Demo**
Compare summaries before and after coreference resolution:

Input Text	BART Summary	SpanBERT → BART Summary
Alice met Bob. She smiled at him.	She smiled.	Alice smiled at Bob.
