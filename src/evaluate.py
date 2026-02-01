import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Load model
from train import DiabetesNN
model = DiabetesNN(X_test.shape[1])
model.load_state_dict(torch.load("model/diabetes_model.pth"))
model.eval()

with torch.no_grad():
    preds = model(X_test)
    predicted_classes = (preds > 0.4).int()

print(confusion_matrix(y_test, predicted_classes))
print(classification_report(y_test, predicted_classes))
