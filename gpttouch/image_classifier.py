import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 加载 embedding
X_rgb = np.load("./embeddings/embeddings_rgb/all_embeddings.npy")     # shape: [N, D]
X_tac = np.load("./embeddings/embeddings_tac/all_embeddings.npy")     # shape: [M, D]

X = np.vstack([X_rgb, X_tac])
y = np.array([0]*len(X_rgb) + [1]*len(X_tac))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))

# ✅ 保存模型
joblib.dump(clf, "tactile_rgb_classifier.pkl")
print("模型已保存为 tactile_rgb_classifier.pkl")
