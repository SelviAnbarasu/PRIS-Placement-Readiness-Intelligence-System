import os
import sys
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

# ============================================================
# PRIS - Rebuild Placement ML Model
# Run this file in the SAME environment where Streamlit runs.
# ============================================================

DATA_PATH = "student_placement_synthetic.csv"
MODEL_PATH = "student_placement_model.pkl"

CATEGORICAL_FEATURES = ["branch", "college_tier"]

TARGET = "placement_status"

DROP_COLUMNS = [
    "placement_status",
    "salary_package_lpa",
]


def main():
    print("=" * 70)
    print("PRIS - Placement ML Model Rebuild")
    print("=" * 70)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"NumPy        : {np.__version__}")
    print(f"joblib       : {joblib.__version__}")
    print()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Keep this script and student_placement_synthetic.csv "
            "in the same project folder."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    required_columns = {
        "branch",
        "college_tier",
        "cgpa",
        "backlogs",
        "coding_skills",
        "dsa_score",
        "aptitude_score",
        "communication_skills",
        "ml_knowledge",
        "system_design",
        "internships",
        "projects_count",
        "certifications",
        "hackathons",
        "open_source_contributions",
        "extracurriculars",
        "placement_status",
        "salary_package_lpa",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing these columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Features / Target
    # --------------------------------------------------------
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    numerical_features = [
        column
        for column in X.columns
        if column not in CATEGORICAL_FEATURES
    ]

    print(f"Features used : {len(X.columns)}")
    print(f"Categorical   : {CATEGORICAL_FEATURES}")
    print(f"Numerical     : {len(numerical_features)}")
    print()

    # --------------------------------------------------------
    # Train / Test Split
    # Same split strategy used in the project
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------
    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features,
            ),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # --------------------------------------------------------
    # Gradient Boosting Model
    # This is the original final model used in PRIS.
    # --------------------------------------------------------
    classifier = GradientBoostingClassifier(
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("Training Gradient Boosting model...")
    model.fit(X_train, y_train)
    print("Training completed.")
    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")
    print()

    # --------------------------------------------------------
    # Save model
    # IMPORTANT: this pickle is created in YOUR current
    # environment, so Streamlit uses the same sklearn version.
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        backup_path = "student_placement_model_old.pkl"
        os.replace(MODEL_PATH, backup_path)
        print(f"Old model backed up as: {backup_path}")

    joblib.dump(model, MODEL_PATH)

    print(f"New model saved as: {MODEL_PATH}")
    print(f"Model size: {os.path.getsize(MODEL_PATH):,} bytes")
    print()

    # --------------------------------------------------------
    # Load test
    # --------------------------------------------------------
    print("Testing saved model...")
    loaded_model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
                "branch": "CSE",
                "college_tier": "Tier-1",
                "cgpa": 8.0,
                "backlogs": 0,
                "coding_skills": 7,
                "dsa_score": 7,
                "aptitude_score": 75,
                "communication_skills": 7,
                "ml_knowledge": 6,
                "system_design": 5,
                "internships": 1,
                "projects_count": 3,
                "certifications": 2,
                "hackathons": 1,
                "open_source_contributions": 1,
                "extracurriculars": 1,
            }
        ]
    )

    sample_prediction = int(loaded_model.predict(sample)[0])
    sample_probability = loaded_model.predict_proba(sample)[0]

    print("Saved model loaded successfully.")
    print(f"Sample prediction : {sample_prediction}")
    print(
        "Sample probability: "
        f"Not Placed={sample_probability[0] * 100:.2f}% | "
        f"Placed={sample_probability[1] * 100:.2f}%"
    )

    print()
    print("=" * 70)
    print("SUCCESS - ML MODEL IS READY FOR STREAMLIT")
    print("=" * 70)


if __name__ == "__main__":
    main()
