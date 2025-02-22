from Model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_classes=2,
                           random_state=42)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Dataset")
plt.savefig('Dataset.png')
plt.show()

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
loss = model.loss(X_test, y_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Loss: ", loss)