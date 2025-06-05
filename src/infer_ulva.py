from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Load model and tokenizer
base_model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./models/ulva-lora-t5")
model.eval()

# Function to run inference
def ask_ulva(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            num_beams=4
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


question = "Are group 2 innate lymphoid cells (ILC2s) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?"
context = (
    "Chronic rhinosinusitis (CRS) is a heterogeneous disease with an uncertain pathogenesis. "
    "Group 2 innate lymphoid cells (ILC2s) represent a recently discovered cell population "
    "implicated in driving Th2 inflammation in CRS. In patients with nasal polyps and eosinophilia, "
    "ILC2s were significantly increased. There was a correlation with tissue eosinophilia and asthma."
)

response = ask_ulva(question, context)
print("🧠 Ulva's Answer:")
print(response)
