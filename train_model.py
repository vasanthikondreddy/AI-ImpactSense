# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("earthquake_alert_balanced_dataset.csv")

# Train model
X = df[["magnitude", "depth", "cdi", "mmi", "sig"]]
y = df["alert"]
model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
