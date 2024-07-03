import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('preprocessed_creditcard.csv')

# Display first few rows to verify data loading
print(data.head())

# Split data into X (features) and y (target)
X = data.drop('Class', axis=1)
y = data['Class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display shapes of train and test sets to verify
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on training set
y_train_pred = model.predict(X_train)

# Make predictions on test set
y_test_pred = model.predict(X_test)

# Classification report on test set
classification_rep = classification_report(y_test, y_test_pred)
print("Classification Report on Test Set:\n", classification_rep)

# ROC AUC Score on test set
roc_auc = roc_auc_score(y_test, y_test_pred)
print(f"ROC AUC Score on Test Set: {roc_auc}")

# Precision-Recall curve on test set
precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')
plt.show()

# Save predictions to CSV files
pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred}).to_csv('train_predictions.csv', index=False)
pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred}).to_csv('test_predictions.csv', index=False)

# Save the ROC curve visualization as well
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve')
plt.savefig('roc_curve.png')
plt.show()

# Save classification report and shapes to a text file
with open('output_summary.txt', 'w') as f:
    f.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\n")
    f.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")
    f.write("\nClassification Report on Test Set:\n")
    f.write(classification_rep)
    f.write(f"\nROC AUC Score on Test Set: {roc_auc}\n")
