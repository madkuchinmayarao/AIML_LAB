

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 10, 100) 
Y = 2 * X + 1 + np.sin(X) + np.random.normal(0, 0.5, 100)  
def locally_weighted_regression(x, X, Y, tau=1.0):
    weights = np.exp(-(X - x) ** 2 / (2 * tau ** 2))
    denominator = np.sum(weights)
    numer = np.sum(weights * Y)
    prediction = numer / denominator
    return prediction

bandwidth = 1.0

# Predict Y values for a range of X values
X_pred = np.linspace(0, 10, 100)
Y_pred = [locally_weighted_regression(x, X, Y, bandwidth) for x in X_pred]

# Create a scatter plot of the original data points
plt.scatter(X, Y, label='Data Points', color='blue')

# Create a line plot for the LWR predictions
plt.plot(X_pred, Y_pred, label="LWR Predictions", color='red', linewidth=5.0)


plt.show()