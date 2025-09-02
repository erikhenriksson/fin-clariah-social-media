import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)

warnings.filterwarnings("ignore")


# Check and setup GPU (CUDA/ROCm)
def setup_gpu():
    """Check GPU availability and setup device (works with both NVIDIA CUDA and AMD ROCm)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available! Device count: {torch.cuda.device_count()}")

        try:
            # This works for both CUDA and ROCm
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        except Exception as e:
            print(f"Could not get detailed GPU info: {e}")
            print("This might be an AMD GPU with ROCm - that's fine!")

        # Check if this is ROCm
        try:
            import torch.version

            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                print(f"ROCm/HIP version: {torch.version.hip}")
            elif hasattr(torch.version, "cuda") and torch.version.cuda is not None:
                print(f"CUDA version: {torch.version.cuda}")
        except:
            pass

        # Clear cache to start fresh
        torch.cuda.empty_cache()
        return device
    else:
        print("No GPU available. Using CPU.")
        return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_and_prepare_data(json_file_path):
    """Load JSON data and prepare for training"""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract texts and labels
    texts = [item["text"] for item in data]
    labels = [str(item["cluster_id"]) for item in data]

    return texts, labels


def get_optimal_batch_size(device, model_size="large"):
    """Suggest optimal batch size based on GPU memory"""
    if device.type == "cpu":
        return 4, 8  # train_batch, eval_batch

    try:
        # Get available GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated()

        # Rough estimates for XLM-R Large
        if model_size == "large":
            if available_memory > 10e9:  # >10GB
                return 16, 32
            elif available_memory > 6e9:  # >6GB
                return 8, 16
            elif available_memory > 4e9:  # >4GB
                return 4, 8
            else:
                return 2, 4
        else:  # base model
            if available_memory > 6e9:
                return 32, 64
            elif available_memory > 4e9:
                return 16, 32
            else:
                return 8, 16

    except Exception as e:
        print(f"Could not determine optimal batch size: {e}")
        return 8, 16  # Safe defaults


def create_classifier(json_file_path):
    """Create and train XLM-R classifier with GPU optimization (CUDA/ROCm)"""

    # Setup GPU
    device = setup_gpu()

    # Load data
    texts, labels = load_and_prepare_data(json_file_path)

    print(f"Loaded {len(texts)} samples")
    print(f"Unique labels: {set(labels)}")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    print(f"Number of classes: {num_labels}")

    # Split data: 70% train, 15% dev, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, encoded_labels, test_size=0.3, random_state=42, stratify=encoded_labels
    )

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

    # Initialize tokenizer and model
    model_name = "xlm-roberta-large"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Move model to device (CRUCIAL!)
    model = model.to(device)
    print(f"Model moved to device: {device}")

    # Get optimal batch sizes
    train_batch_size, eval_batch_size = get_optimal_batch_size(device, "large")
    print(f"Using batch sizes - Train: {train_batch_size}, Eval: {eval_batch_size}")

    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    dev_dataset = TextDataset(X_dev, y_dev, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments - optimized for GPU (CUDA/ROCm)
    use_gpu = device.type == "cuda"
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=1,  # Adjust if memory issues
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        seed=42,
        fp16=use_gpu,  # Enable mixed precision for GPU
        dataloader_num_workers=4 if use_gpu else 0,  # Parallel data loading
        remove_unused_columns=False,
        dataloader_pin_memory=use_gpu,  # Pin memory for faster transfer
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    print(f"Training on device: {next(model.parameters()).device}")

    # Monitor GPU usage during training
    if use_gpu:
        print(
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

    trainer.train()

    if use_gpu:
        print(
            f"GPU memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Loss: {test_results['eval_loss']:.4f}")

    # Get predictions for classification report
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    # Convert back to original labels for report
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Print detailed classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test_labels, y_pred_labels))

    print(f"\nOverall Accuracy: {accuracy_score(y_test_labels, y_pred_labels):.4f}")

    return model, tokenizer, label_encoder, device


def predict_text(text, model, tokenizer, label_encoder, device):
    """Predict text with GPU support (CUDA/ROCm)"""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
        confidence = predictions[0][predicted_class_id].item()

    return predicted_label, confidence


# Example usage:
if __name__ == "__main__":
    json_file_path = "clusters_final_final/fi_embeds_NA-NB-OP/clustered_data.json"

    try:
        model, tokenizer, label_encoder, device = create_classifier(json_file_path)
        print("Training completed successfully!")

        # Test prediction
        sample_text = "Your sample text here"
        pred_label, confidence = predict_text(
            sample_text, model, tokenizer, label_encoder, device
        )
        print(f"\nSample prediction: '{pred_label}' (confidence: {confidence:.3f})")

        # Clean up GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print("GPU memory cleared.")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{json_file_path}'")
        print("Please make sure the file path is correct and the file exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
