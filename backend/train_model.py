import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib   # use joblib instead of pickle

# 1. Load your training dataset
df = pd.read_csv("Cleaned_dataset.csv")

# 2. Define features and target column
X = df[["magnitude", "depth", "cdi", "mmi", "sig"]]
y = df["alert"]

# 3. Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# 4. Save the trained model to a .pkl file using joblib
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl using joblib")
