from datasets import load_dataset
from transformers import AutoTokenizer

# Load PubMedQA dataset
dataset = load_dataset("pubmed_qa", "pqa_artificial")
train_data = dataset["train"]

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Map 'yes', 'no', 'maybe' to full sentences
def convert_label_to_text(label):
    if label == "yes":
        return "The answer is yes."
    elif label == "no":
        return "The answer is no."
    else:
        return "The answer is maybe."

def preprocess(example):
    question = example.get("question", "")
    raw_context = example.get("context", "")
    if isinstance(raw_context, dict) and "contexts" in raw_context:
        context = " ".join(raw_context["contexts"])
    else:
        context = str(raw_context)

    label = example.get("final_decision", "")

   
    question = str(question) if question else ""
    context = str(context) if context else ""
    answer_text = convert_label_to_text(label)

    # Tokenize question + context as input
    tokenized = tokenizer(
        question,
        context,
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Tokenize answer as output label
    labels = tokenizer(
        answer_text,
        padding="max_length",
        truncation=True,
        max_length=16
    )

    tokenized["labels"] = labels["input_ids"]
    return tokenized


# Apply preprocessing to the full dataset
tokenized_dataset = train_data.map(preprocess)

# Save for later use 
tokenized_dataset.save_to_disk("data/preprocessed_pubmedqa")
print("✅ Tokenized and saved!")
