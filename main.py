from src.data_loader import load_data, pre_processing_split
from src.preprocessing import create_preprocessor, features_scaling
from src.model import build_model, train_model
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np

data = load_data('data/creditcard.csv')
X_train, X_test, y_train, y_test = pre_processing_split(data)
ct = create_preprocessor(X_train)
X_train_scaled, X_test_scaled = features_scaling(X_train, X_test, ct)
model = build_model(X_train_scaled)
history = train_model(model, X_train_scaled, y_train)

y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy Score: {acc}')
print(f'Recall Score: {recall}')
print('Confusion Matrix')
print(cm)



