import matplotlib.pyplot as plt
# Evaluation loss and training loss over epochs
training_loss = [0.17398814957936604]  # Final training loss at the end of epoch 3
eval_loss = [0.2640794813632965, 0.2353651225566864, 0.3019926846027374]
epochs = [1, 2, 3]

# Plot training vs evaluation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, [training_loss[0]]*3, label="Training Loss", linestyle='--', marker='o')
plt.plot(epochs, eval_loss, label="Evaluation Loss", marker='o')
plt.title("Training vs. Evaluation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

