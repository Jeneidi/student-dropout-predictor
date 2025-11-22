import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessData import preprocess_data

def train_baseline_model():
    df = preprocess_data()

    X = df.drop(columns=["dropout_risk"])
    y = df["dropout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Baseline Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    train_baseline_model()
