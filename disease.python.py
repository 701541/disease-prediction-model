import numpy as np
from spipklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Fill NaNs with 'None' for simplicity
dataset.fillna("None", inplace=True)

# Combine all symptoms into a single set to one-hot encode
all_symptoms = set()
for col in dataset.columns[1:]:
    all_symptoms.update(dataset[col].unique())
all_symptoms.discard("None")
all_symptoms = sorted(list(all_symptoms))

# Create binary (or severity-based) feature vectors for each record
def encode_symptoms(row):n
    symptoms_present = set(row[1:])
    return [1 if symptom in symptoms_present else 0 for symptom in all_symptoms]

X = dataset.apply(encode_symptoms, axis=1, result_type="expand")
X.columns = all_symptoms

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(dataset["Disease"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

accuracy, report
