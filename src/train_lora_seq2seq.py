
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,  
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load pre-tokenized dataset
dataset = load_from_disk("data/preprocessed_pubmedqa")
dataset = dataset.select(range(50)) 


# Use T5 small for generation
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],  
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Modify dataset format: convert input_ids/labels to T5-style input-output strings
def preprocess_t5_format(example):
    input_text = f"question: {example['question']} context: {' '.join(example['context']['contexts'])}" if isinstance(example["context"], dict) else f"question: {example['question']} context: {example['context']}"
    label_text = example["final_decision"]
    label_text = {
        "yes": "The answer is yes.",
        "no": "The answer is no.",
        "maybe": "The answer is maybe."
    }.get(label_text.lower(), "The answer is maybe.")
    model_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(label_text, truncation=True, padding="max_length", max_length=16)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_t5_format)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/ulva-lora-t5",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

model.save_pretrained("./models/ulva-lora-t5")
