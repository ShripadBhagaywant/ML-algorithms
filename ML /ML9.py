#Random Forest.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(0)

# Generate moon-shaped dataset
x, y = make_moons(500, noise=0.1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.20, random_state=13)

# Create the plot
plt.figure(figsize=(12,8))

# Scatter plot of training data
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=50)

# Add labels and title
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Random Training Data')

# Show the plot
plt.show()