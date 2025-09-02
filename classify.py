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

    # Handle preds - assuming it's a list but we want the first element
    # or if it's meant to be cluster_id, adjust accordingly
    if "preds" in data[0] and data[0]["preds"]:
        labels = [item["preds"][0] if item["preds"] else "UNKNOWN" for item in data]
    else:
        # Fallback to cluster_id if preds is empty
        labels = [str(item["cluster_id"]) for item in data]

    return texts, labels


def create_classifier(json_file_path):
    """Create and train XLM-R classifier"""

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

    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    dev_dataset = TextDataset(X_dev, y_dev, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments - optimized for simplicity and effectiveness
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
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
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
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
    trainer.train()

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

    return model, tokenizer, label_encoder


# Example usage:
if __name__ == "__main__":
    # Replace 'your_data.json' with your actual file path
    json_file_path = "clusters_final_final/fi_embeds_NA-NB-OP/clustered_data.json"

    try:
        model, tokenizer, label_encoder = create_classifier(json_file_path)
        print("Training completed successfully!")

        # Example prediction on new text
        def predict_text(text, model, tokenizer, label_encoder):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                predicted_label = label_encoder.inverse_transform([predicted_class_id])[
                    0
                ]
                confidence = predictions[0][predicted_class_id].item()

            return predicted_label, confidence

        # Test prediction
        sample_text = "Your sample text here"
        pred_label, confidence = predict_text(
            sample_text, model, tokenizer, label_encoder
        )
        print(f"\nSample prediction: '{pred_label}' (confidence: {confidence:.3f})")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{json_file_path}'")
        print("Please make sure the file path is correct and the file exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
