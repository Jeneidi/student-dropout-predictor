import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from preprocessData import preprocess_data

def train_smote_model():
    df = preprocess_data()

    X = df.drop(columns=["dropout_risk"])
    y = df["dropout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to ONLY the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", y_train_resampled.value_counts())

    model = LogisticRegression(max_iter=2000)

    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    print("SMOTE + Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    train_smote_model()
