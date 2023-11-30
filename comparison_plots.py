import matplotlib.pyplot as plt
import pandas as pd

# Load the csv file
df = pd.read_csv('model_logs/torch128_2.csv')

# Calculate the moving averages
df['length_mean_ma'] = df['length_mean'].rolling(20).mean()
df['loss_ma'] = df['loss'].rolling(20).mean()

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Length Mean Plot
axs[0].set_title('Snake Mean Length vs Iteration (Batch Size 128)')
axs[0].plot(df['iteration'], df['length_mean'], label='Length Mean', color='skyblue')
axs[0].plot(df['iteration'], df['length_mean_ma'], label='Length Mean Moving Average', color='blue')
axs[0].set_ylabel('Mean Length')
axs[0].set_xlabel('Iteration')
axs[0].legend()

# Loss Plot
axs[1].set_title('Loss vs Iteration (Batch Size 128)')
axs[1].plot(df['iteration'], df['loss'], label='Loss', color='orange')
axs[1].plot(df['iteration'], df['loss_ma'], label='Loss Moving Average', color='red')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Iteration')
axs[1].legend()

plt.tight_layout()

# Save the plot to a file
plt.savefig('images/raining_performance_plot.png', format='png')

# Show the plot
plt.show()
