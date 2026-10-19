"""Train and evaluate a simple student-query text classifier."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_queries.csv"

def main():
    df = pd.read_csv(DATA)
    if df["query"].isna().any() or df["label"].isna().any():
        raise ValueError("Dataset contains missing query or label values.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["query"], df["label"], test_size=0.20, random_state=42,
        stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                  strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(df["label"].unique())
    print(f"Dataset rows: {len(df)}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"Macro precision: {precision_score(y_test, pred, average='macro', zero_division=0):.3f}")
    print(f"Macro recall: {recall_score(y_test, pred, average='macro', zero_division=0):.3f}")
    print(f"Macro F1: {f1_score(y_test, pred, average='macro', zero_division=0):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; label order shown):")
    print(labels)
    print(confusion_matrix(y_test, pred, labels=labels))
    print("\nNote: synthetic demo data only; metrics are not evidence of real-world performance.")

    # Demonstrate predictions on new sample queries, not included in the dataset.
    samples = [
        "I need a copy of my semester payment receipt",
        "My password reset email is not arriving",
        "When is the practical exam scheduled?"
    ]
    print("\nSample predictions:")
    for query, label in zip(samples, model.predict(samples)):
        print(f"- {query} -> {label}")

if __name__ == "__main__":
    main()
