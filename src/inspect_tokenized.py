from datasets import load_from_disk
from transformers import AutoTokenizer

# Load the saved dataset
dataset = load_from_disk("data/preprocessed_pubmedqa")

# Load tokenizer for decoding tokens back to text
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Print the first sample
sample = dataset[0]

print("\n Raw Tokenized Data:")
for key, value in sample.items():
    if isinstance(value, list):
        print(f"{key}: {value[:10]}...")
    else:
        print(f"{key}: {value}")
 
# Decode input_ids and labels to human-readable text
input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
label_text = tokenizer.decode(sample["labels"], skip_special_tokens=True)

print("\n Decoded Input Text:")
print(input_text)

print("\n Decoded Label (Expected Answer):")
print(label_text)
