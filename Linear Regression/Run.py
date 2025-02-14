import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Data
fig = plt.figure(figsize=(6, 6), dpi=100)
plt.scatter(X_train, y_train, color='blue', label="Train Data")
plt.scatter(X_test, y_test, color='red', label="Test Data")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.title("Train & Test Data Distribution")
plt.savefig("Data.png")
plt.show()

from Model import LinearRegression

train = LinearRegression()
train.fit(X_train, y_train)

predicted_test = train.predict(X_test)
predicted_all = train.predict(X)

# Fit Data
fig2 = plt.figure(figsize=(6, 6), dpi=100)
plt.scatter(X_train, y_train, color='blue', label="Train Data")
plt.scatter(X_test, y_test, color='red', label="Test Data")
plt.plot(X, predicted_all, color='green', lw=2, label="Regression Line")  
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.title("Regression Line with data")
plt.savefig("Fit_Data.png")
plt.show()


print("Weight: ", train.weights)
print("Bias: ", train.bias)

def MSE(y, y_hat):
    return np.mean((y - y_hat) ** 2)

print("MSE: ", MSE(y_test, predicted_test))