import matplotlib.pyplot as plt
# Evaluation loss values (from your logs)
eval_loss = [0.2640794813632965, 0.2353651225566864, 0.3019926846027374]
eval_epochs = [1, 2, 3]

# Plot evaluation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(eval_epochs, eval_loss, marker='o', color='orange')
plt.title("Evaluation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Evaluation Loss")
plt.grid(True)
plt.show()
