import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure folder for saving plots exists
os.makedirs("static", exist_ok=True)

data = pd.read_csv("disease_data.csv")

# Save correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("static/heatmap.png")
plt.close()

X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class DecisionTree:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y):
        if len(set(y)) == 1:
            self.label = y[0]
            return
        if self.depth >= self.max_depth or len(y) <= 1:
            self.label = np.round(np.mean(y))
            return

        best_gini = float("inf")
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left = X[:, feature_index] <= threshold
                right = X[:, feature_index] > threshold
                if len(y[left]) == 0 or len(y[right]) == 0:
                    continue
                gini = self._gini(y[left], y[right])
                if gini < best_gini:
                    best_gini = gini
                    self.feature_index = feature_index
                    self.threshold = threshold

        if self.feature_index is None:
            self.label = np.round(np.mean(y))
            return

        left = X[:, self.feature_index] <= self.threshold
        right = X[:, self.feature_index] > self.threshold
        self.left = DecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.right = DecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.left.fit(X[left], y[left])
        self.right.fit(X[right], y[right])

    def _gini(self, left_y, right_y):
        def gini_impurity(y):
            classes, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

        total = len(left_y) + len(right_y)
        return (len(left_y)/total)*gini_impurity(left_y) + (len(right_y)/total)*gini_impurity(right_y)

    def predict(self, x):
        if self.label is not None:
            return self.label
        if x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict_batch(X_test)
accuracy = np.mean(y_pred == y_test)

# Save accuracy and confusion matrix
def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix

conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, index=["No Disease", "Disease"], columns=["No Disease", "Disease"])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix.png")
plt.close()

# Export values for Flask
model_accuracy = accuracy
