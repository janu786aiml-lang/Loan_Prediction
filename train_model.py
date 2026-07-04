"""
==============================================================
Loan Prediction System
Machine Learning Model Training

Description:
    This module trains multiple Machine Learning models
    for Loan Eligibility Prediction and automatically
    selects the best performing model.

Author : Your Name
Version : 1.0
==============================================================
"""

# ==========================================================
# Standard Library Imports
# ==========================================================

import logging
import os
import warnings

# ==========================================================
# Third Party Imports
# ==========================================================

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification Models

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

# Evaluation Metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Local Imports
# ==========================================================

import config

# ==========================================================
# Ignore Warnings
# ==========================================================

warnings.filterwarnings("ignore")

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)

# ==========================================================
# Loan Model Trainer
# ==========================================================

class LoanModelTrainer:
    """
    Complete Machine Learning Pipeline

    1. Load Dataset

    2. Validate Dataset

    3. Feature Selection

    4. Train Test Split

    5. Feature Scaling

    6. Train Models

    7. Evaluate Models

    8. Select Best Model

    9. Save Model
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("INITIALIZING MODEL TRAINER")
        logger.info("=" * 60)

        # -------------------------
        # Dataset
        # -------------------------

        self.df = None

        # -------------------------
        # Features & Target
        # -------------------------

        self.X = None
        self.y = None

        # -------------------------
        # Train Test Data
        # -------------------------

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        # -------------------------
        # Scaler
        # -------------------------

        self.scaler = StandardScaler()

        # -------------------------
        # Models
        # -------------------------

        self.models = {}

        # -------------------------
        # Results
        # -------------------------

        self.results = {}

        self.best_model = None

        self.best_model_name = None

        self.best_accuracy = 0.0

        logger.info("Trainer Initialized Successfully.")

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        logger.info("=" * 60)
        logger.info("LOADING ENCODED DATASET")
        logger.info("=" * 60)

        if not os.path.exists(config.ENCODED_DATASET_PATH):

            raise FileNotFoundError(

                f"\nEncoded dataset not found\n"

                f"{config.ENCODED_DATASET_PATH}\n\n"

                f"Run preprocess.py first."

            )

        self.df = pd.read_csv(
            config.ENCODED_DATASET_PATH
        )

        logger.info(
            "Dataset Loaded Successfully."
        )

        logger.info(
            f"Rows : {self.df.shape[0]}"
        )

        logger.info(
            f"Columns : {self.df.shape[1]}"
        )

        logger.info(
            f"Memory Usage : "
            f"{round(self.df.memory_usage(deep=True).sum()/1024,2)} KB"
        )

    # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_dataset(self):

        logger.info("=" * 60)
        logger.info("VALIDATING DATASET")
        logger.info("=" * 60)

        if self.df.empty:

            raise ValueError(
                "Dataset is Empty."
            )

        if "Loan_Status" not in self.df.columns:

            raise ValueError(
                "Loan_Status column missing."
            )

        logger.info(
            "Target Column Found."
        )

        logger.info(
            "Checking Missing Values..."
        )

        missing = self.df.isnull().sum().sum()

        if missing > 0:

            raise ValueError(

                f"Dataset still contains "

                f"{missing} missing values.\n"

                f"Run preprocess.py again."

            )

        logger.info(
            "No Missing Values Found."
        )

        logger.info(
            "Dataset Validation Successful."
        )

    # ======================================================
    # Dataset Overview
    # ======================================================

    def dataset_summary(self):

        logger.info("=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)

        print("\n")

        print(self.df.head())

        print("\n")

        print(self.df.describe())

        print("\nColumns")

        print(self.df.columns.tolist())

        print("\nTarget Distribution")

        print(self.df["Loan_Status"].value_counts())

    # ======================================================
    # Feature Selection
    # ======================================================

    def feature_selection(self):

        logger.info("=" * 60)
        logger.info("FEATURE SELECTION")
        logger.info("=" * 60)

        # Remove Loan_ID (Not useful for prediction)
        if "Loan_ID" in self.df.columns:

            self.df.drop(
                columns=["Loan_ID"],
                inplace=True
            )

            logger.info("Loan_ID column removed.")

        # Features
        self.X = self.df.drop(
            columns=["Loan_Status"]
        )

        # Target
        self.y = self.df["Loan_Status"]

        logger.info(f"Number of Features : {self.X.shape[1]}")
        logger.info(f"Training Samples   : {self.X.shape[0]}")

        logger.info("\nFeature Names:")

        for column in self.X.columns:
            logger.info(f"   • {column}")

        logger.info("Feature Selection Completed.")

    # ======================================================
    # Train Test Split
    # ======================================================

    def split_dataset(self):

        logger.info("=" * 60)
        logger.info("TRAIN TEST SPLIT")
        logger.info("=" * 60)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(

            self.X,

            self.y,

            test_size=config.TEST_SIZE,

            random_state=config.RANDOM_STATE,

            stratify=self.y

        )

        logger.info(
            f"Training Records : {len(self.X_train)}"
        )

        logger.info(
            f"Testing Records : {len(self.X_test)}"
        )

        logger.info("Dataset Split Successful.")

    # ======================================================
    # Feature Scaling
    # ======================================================

    def scale_features(self):

        logger.info("=" * 60)
        logger.info("FEATURE SCALING")
        logger.info("=" * 60)

        numerical_columns = [

            "ApplicantIncome",

            "CoapplicantIncome",

            "LoanAmount",

            "Loan_Amount_Term",

            "Credit_History"

        ]

        logger.info("Scaling Numerical Features...")

        self.X_train[numerical_columns] = self.scaler.fit_transform(

            self.X_train[numerical_columns]

        )

        self.X_test[numerical_columns] = self.scaler.transform(

            self.X_test[numerical_columns]

        )

        logger.info("Feature Scaling Completed.")

    # ======================================================
    # Save Scaler
    # ======================================================

    def save_scaler(self):

        logger.info("=" * 60)
        logger.info("SAVING SCALER")
        logger.info("=" * 60)

        joblib.dump(

            self.scaler,

            config.SCALER_PATH

        )

        logger.info(

            f"Scaler Saved Successfully.\n"

            f"{config.SCALER_PATH}"

        )

    # ======================================================
    # Dataset Preparation Pipeline
    # ======================================================

    def prepare_dataset(self):

        logger.info("=" * 60)
        logger.info("PREPARING DATASET")
        logger.info("=" * 60)

        self.load_dataset()

        self.validate_dataset()

        self.dataset_summary()

        self.feature_selection()

        self.split_dataset()

        self.scale_features()

        self.save_scaler()

        logger.info("=" * 60)
        logger.info("DATASET PREPARATION COMPLETED")
        logger.info("=" * 60)

    # ======================================================
    # Train Decision Tree
    # ======================================================

    def train_decision_tree(self):

        logger.info("=" * 60)
        logger.info("TRAINING DECISION TREE")
        logger.info("=" * 60)

        model = DecisionTreeClassifier(
            random_state=config.RANDOM_STATE,
            max_depth=5
        )

        model.fit(self.X_train, self.y_train)

        self.models["Decision Tree"] = model

        logger.info("Decision Tree Trained Successfully.")

    # ======================================================
    # Train Random Forest
    # ======================================================

    def train_random_forest(self):

        logger.info("=" * 60)
        logger.info("TRAINING RANDOM FOREST")
        logger.info("=" * 60)

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_STATE,
            max_depth=8
        )

        model.fit(self.X_train, self.y_train)

        self.models["Random Forest"] = model

        logger.info("Random Forest Trained Successfully.")

    # ======================================================
    # Train KNN
    # ======================================================

    def train_knn(self):

        logger.info("=" * 60)
        logger.info("TRAINING KNN")
        logger.info("=" * 60)

        model = KNeighborsClassifier(
            n_neighbors=7
        )

        model.fit(self.X_train, self.y_train)

        self.models["KNN"] = model

        logger.info("KNN Trained Successfully.")

    # ======================================================
    # Train XGBoost
    # ======================================================

    def train_xgboost(self):

        logger.info("=" * 60)
        logger.info("TRAINING XGBOOST")
        logger.info("=" * 60)

        model = XGBClassifier(

            n_estimators=100,

            learning_rate=0.1,

            max_depth=4,

            random_state=config.RANDOM_STATE,

            eval_metric="logloss"

        )

        model.fit(
            self.X_train,
            self.y_train
        )

        self.models["XGBoost"] = model

        logger.info("XGBoost Trained Successfully.")

    # ======================================================
    # Train All Models
    # ======================================================

    def train_models(self):

        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)

        self.train_decision_tree()

        self.train_random_forest()

        self.train_knn()

        self.train_xgboost()

        logger.info("=" * 60)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 60)

            # ======================================================
    # Evaluate Models
    # ======================================================

    def evaluate_models(self):

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)

        for model_name, model in self.models.items():

            logger.info(f"Evaluating {model_name}...")

            predictions = model.predict(self.X_test)

            accuracy = accuracy_score(
                self.y_test,
                predictions
            )

            precision = precision_score(
                self.y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                self.y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                self.y_test,
                predictions,
                zero_division=0
            )

            confusion = confusion_matrix(
                self.y_test,
                predictions
            )

            report = classification_report(
                self.y_test,
                predictions,
                zero_division=0
            )

            self.results[model_name] = {

                "Model": model_name,

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1 Score": f1,

                "Confusion Matrix": confusion,

                "Classification Report": report

            }

            logger.info(
                f"{model_name} Accuracy : {accuracy:.4f}"
            )

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION COMPLETED")
        logger.info("=" * 60)

    # ======================================================
    # Compare Models
    # ======================================================

    def compare_models(self):

        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)

        comparison = []

        for name, result in self.results.items():

            comparison.append({

                "Model": name,

                "Accuracy": round(result["Accuracy"], 4),

                "Precision": round(result["Precision"], 4),

                "Recall": round(result["Recall"], 4),

                "F1 Score": round(result["F1 Score"], 4)

            })

        comparison_df = pd.DataFrame(comparison)

        comparison_df = comparison_df.sort_values(

            by="Accuracy",

            ascending=False

        )

        print("\n")
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        best = comparison_df.iloc[0]

        self.best_model_name = best["Model"]

        self.best_accuracy = best["Accuracy"]

        self.best_model = self.models[
            self.best_model_name
        ]

        logger.info(
            f"Best Model : {self.best_model_name}"
        )

        logger.info(
            f"Best Accuracy : {self.best_accuracy:.4f}"
        )

    # ======================================================
    # Save Best Model
    # ======================================================

    def save_best_model(self):

        logger.info("=" * 60)
        logger.info("SAVING BEST MODEL")
        logger.info("=" * 60)

        joblib.dump(

            self.best_model,

            config.MODEL_PATH

        )

        logger.info(

            f"Best Model Saved Successfully\n"

            f"{config.MODEL_PATH}"

        )

    # ======================================================
    # Complete Training Pipeline
    # ======================================================

    def run_pipeline(self):

        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        self.prepare_dataset()

        self.train_models()

        self.evaluate_models()

        self.compare_models()

        self.save_best_model()

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 60)

        # ======================================================
# Main Function
# ======================================================

def main():

    trainer = LoanModelTrainer()

    trainer.run_pipeline()


if __name__ == "__main__":

    main()