import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up Trainer for fine-tuning
training_args = TrainingArguments(
    output_dir="./results",               # Output directory for model checkpoints
    evaluation_strategy="epoch",          # Evaluate at the end of every epoch
    learning_rate=2e-5,                   # Learning rate (can be adjusted if needed)
    per_device_train_batch_size=8,        # Batch size for training
    per_device_eval_batch_size=16,        # Batch size for evaluation
    num_train_epochs=3,                   # Number of training epochs (adjust as needed)
    weight_decay=0.01,                    # Weight decay to reduce overfitting
    logging_dir='./logs',                 # Directory for logging
    logging_steps=100,                    # Log every 100 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Train the model (this will fine-tune the classifier layer)
trainer.train()

model.save_pretrained("./fine_tuned_model_v1")  # Save model to local directory
tokenizer.save_pretrained("./fine_tuned_model_v1")

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Evaluation results: {eval_result}")

# Make predictions on the test set
predictions = trainer.predict(tokenized_datasets["test"])
print(f"Predictions: {predictions.predictions}")

model = BertForSequenceClassification.from_pretrained("fine_tuned_model_v1")
tokenizer = BertTokenizer.from_pretrained("fine_tuned_model_v1")

# Use the model for inference (example)
review = "This movie was fantastic!"
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
sentiment = "positive" if prediction == 1 else "negative"
print(f"Sentiment: {sentiment}")



