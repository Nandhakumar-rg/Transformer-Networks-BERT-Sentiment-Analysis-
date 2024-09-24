import matplotlib.pyplot as plt

# Example data (replace with the loss values from your logs)
epochs = [0.03, 0.06, 0.1, 0.13, 0.16, 0.19, 0.22, 0.26, 0.29, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.51, 0.54, 0.58, 0.61, 0.64, 0.67, 0.7, 0.74, 0.77, 0.8, 0.83, 0.86, 0.9, 0.93, 0.96, 0.99]
loss_values = [0.5808, 0.3581, 0.3056, 0.3375, 0.3258, 0.3196, 0.3365, 0.264, 0.306, 0.3438, 0.2804, 0.3036, 0.3065, 0.2783, 0.296, 0.2561, 0.2936, 0.2378, 0.3171, 0.298, 0.2627, 0.2903, 0.2673, 0.2769, 0.2352, 0.2603, 0.2302, 0.26, 0.259, 0.2657, 0.2525]

# Plot training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o')
plt.title("Training Loss Over Time")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
