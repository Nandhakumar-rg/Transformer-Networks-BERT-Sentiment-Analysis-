import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the fine-tuned model (replace with your actual model path)
model = BertForSequenceClassification.from_pretrained("fine_tuned_model_v1")

# Set up the Trainer for prediction (you can skip the training part if you've already trained the model)
trainer = Trainer(
    model=model
)

# Get predictions from the test set
predictions = trainer.predict(tokenized_datasets["test"])
predictions_logits = predictions.predictions  # These are the raw logits
predicted_labels = np.argmax(predictions_logits, axis=1)

# Extract true labels from the test set
true_labels = np.array(tokenized_datasets["test"]["label"])

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
