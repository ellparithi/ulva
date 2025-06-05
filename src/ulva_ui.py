


import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Load base + LoRA model
BASE_MODEL = "t5-small"
LORA_PATH = "./models/ulva-lora-t5"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Ulva: Healthcare Q&A Mini UI")

with st.form("ulva_form"):
    question = st.text_input("Enter your question")
    context = st.text_area("Paste the context (e.g. abstract, notes, paper section)")
    submitted = st.form_submit_button("Get Answer")

if submitted:
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    with st.spinner("Ulva is thinking..."):
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    st.markdown("### Ulva's Answer:")
    st.success(decoded)
