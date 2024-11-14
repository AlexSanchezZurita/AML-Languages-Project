import os
import pandas as pd
from datasets import load_dataset, Dataset, Audio
from transformers import Wav2Vec2Processor, HubertModel, TrainingArguments, Trainer
import torch
import random


# Load the dataset path
dataset_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\4th year\Advanced Machine Learning\project\cv-corpus-18.0-delta-2024-06-14-ca\cv-corpus-18.0-delta-2024-06-14\ca\wav"

# Load TSV files for Common Voice
validated_tsv_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\4th year\Advanced Machine Learning\project\cv-corpus-18.0-delta-2024-06-14-ca\cv-corpus-18.0-delta-2024-06-14\ca\validated.tsv"

# Change the extension of each file in validated_tsv_path to .wav
validated_df = pd.read_csv(validated_tsv_path, sep='\t')
validated_df['path'] = validated_df['path'].apply(lambda x: os.path.splitext(x)[0] + '.wav')

# Correct paths to point to the full audio file paths
validated_df['path'] = validated_df['path'].apply(lambda x: os.path.join(dataset_path, x))

# Create Hugging Face dataset from the data frame
dataset = Dataset.from_pandas(validated_df)

# Add audio column
dataset = dataset.cast_column("path", Audio())

# Load HuBERT processor and model
model_name = "facebook/hubert-large-ls960-ft"  # Using a pre-trained HuBERT model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)

# Self-Supervised Preprocessing Function: Add Masking
def preprocess_function(batch, mask_prob=0.15):
    # Load audio and prepare input features
    audio = batch["path"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    input_values = inputs.input_values[0]

    # Masking: Mask a percentage of the input values
    mask = torch.rand(input_values.shape) < mask_prob
    masked_input_values = input_values.clone()
    masked_input_values[mask] = 0  # Set masked values to zero

    batch["input_values"] = masked_input_values
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["original_values"] = input_values  # Store original values for reconstruction
    return batch

# Apply the preprocess function to the dataset
encoded_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=1)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./hubert-ssl-finetuned-catalan",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=3e-4,
    num_train_epochs=10,
    warmup_steps=500,
    save_total_limit=2,
    fp16=False,  # Turn true when GPU available
)

# Custom Data Collator for Self-Supervised Learning
from typing import Any, Dict, List, Union
class CustomDataCollatorSSL:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = torch.stack([feature["input_values"] for feature in features])
        original_values = torch.stack([feature["original_values"] for feature in features])
        attention_mask = torch.stack([feature["attention_mask"] for feature in features])

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "original_values": original_values,
        }

# Use the custom data collator
data_collator = CustomDataCollatorSSL()

# Define a custom loss function for self-supervised learning
def compute_loss(model, inputs):
    # Forward pass
    outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"])

    # Compute reconstruction loss: Mean Squared Error between masked input and original values
    reconstructed_values = outputs.last_hidden_state
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(reconstructed_values, inputs["original_values"])

    return loss

# Custom Trainer for Self-Supervised Learning
from transformers import Trainer
class SSLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

# Trainer initialization
trainer = SSLTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,  # You can split the dataset for evaluation separately
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

# Fine-tune the model using self-supervised learning
trainer.train()
