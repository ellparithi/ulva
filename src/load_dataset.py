from datasets import load_dataset

# PubMedQA dataset from Hugging Face
dataset = load_dataset("pubmed_qa", "pqa_artificial")

# Check what splits it has (train/test)
print("Dataset splits:", dataset.keys())

# A single sample
print("\nSample entry:")
print(dataset["train"][0])
