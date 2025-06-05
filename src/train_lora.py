from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load dataset
dataset = load_from_disk("data/preprocessed_pubmedqa")

# Load tokenizer and base model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=128)  

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Training configuration
training_args = TrainingArguments(
    output_dir="./models/ulva-lora",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no",  
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),  
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train!
trainer.train()
