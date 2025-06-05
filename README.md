# Ulva: Lightweight Healthcare QA Mini-LLM

Ulva is a fine-tuned domain-specific MiniLLM designed for real-time healthcare question answering. Built on top of `t5-small`, it uses LoRA and PEFT techniques to specialize on medical QA datasets like MedQuAD, offering fast and relevant answers in resource-constrained environments.

---

## Features

- Domain Adaptation: Specialized on real-world medical questions via MedQuAD.
- Lightweight: Based on `t5-small` with LoRA adapters for low-resource fine-tuning.
- Modular: Separated training pipeline and inference app using Streamlit.
- Benchmarkable: Compare Ulva’s responses side-by-side with GPT-3.5 outputs.
- Built for Experimentation: Swap datasets, adapters, and model architectures easily.

---

## Tech Stack

- Model: `t5-small` (Hugging Face Transformers)
- Fine-Tuning: LoRA, PEFT
- Dataset: MedQuAD
- Frameworks: Hugging Face, PyTorch, Scikit-learn
- Interface: Streamlit
- Evaluation: Side-by-side comparison vs GPT-3.5 (via OpenAI API)

---

## Project Structure

```
ulva/
├── data/               # Preprocessed MedQuAD or other datasets
├── src/                # All training and inference code
│   ├── train_model.py  # LoRA fine-tuning script
│   └── app.py          # Streamlit-based inference UI
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ellparithi/ulva.git
cd ulva
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run src/app.py
```

---

## Evaluation & Comparison

You can run `app.py` to:
- Ask a medical question (e.g., "What are the symptoms of dengue?")
- See how Ulva answers using the fine-tuned T5 model
- Compare it to GPT-3.5's answer side-by-side (requires OpenAI API key)

---

## Example Use Case

**Question:** “Can asthma be cured in children?”  
**Ulva Answer:** “Asthma has no known cure but can often be managed with treatment.”  
**GPT-3.5 Answer:** “Asthma is a chronic condition, but with medication and care, symptoms in children can be controlled.”

---

## Future Work

- Add support for more medical datasets (PubMedQA, HealthCareMagic)
- Deploy Ulva as an API (FastAPI or Hugging Face Space)
- Evaluate hallucination rate vs large models
- Add attention-based explainability

---

## License

MIT License. You are free to use, modify, and build on Ulva.

---

## Author

Created by [@ellparithi](https://github.com/ellparithi)  
Open to contributions, suggestions, and collaborations.
