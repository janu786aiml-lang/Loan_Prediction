"""
=============================================================
Loan Prediction System
Data Preprocessing and Exploratory Data Analysis

Author : Your Name
Project: Loan Prediction using Machine Learning
=============================================================
"""

import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import config


# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Loan Data Preprocessor
# ============================================================

class LoanDataPreprocessor:
    """
    Performs

    • Data Loading
    • Data Analysis
    • Report Generation
    • Visualization
    • Data Cleaning
    • Label Encoding
    """

    def __init__(self):

        self.df = None
        self.label_encoders = {}

    # ========================================================
    # Load Dataset
    # ========================================================

    def load_dataset(self):

        logger.info("Loading dataset...")

        if not os.path.exists(config.DATASET_PATH):
            raise FileNotFoundError(
                f"Dataset not found:\n{config.DATASET_PATH}"
            )

        self.df = pd.read_csv(config.DATASET_PATH)

        logger.info("Dataset Loaded Successfully.")
        logger.info(f"Shape : {self.df.shape}")

        return self.df

    # ========================================================
    # Dataset Information
    # ========================================================

    def dataset_information(self):

        logger.info("=" * 60)
        logger.info("DATASET INFORMATION")
        logger.info("=" * 60)

        print("\nShape")
        print(self.df.shape)

        print("\nColumns")
        print(self.df.columns.tolist())

        print("\nData Types")
        print(self.df.dtypes)

        print("\nFirst Five Rows")
        print(self.df.head())

        print("\nLast Five Rows")
        print(self.df.tail())

    # ========================================================
    # Missing Values
    # ========================================================

    def missing_values(self):

        logger.info("=" * 60)
        logger.info("MISSING VALUES")
        logger.info("=" * 60)

        missing = self.df.isnull().sum()

        print("\nMissing Values\n")
        print(missing)

        return missing

    # ========================================================
    # Duplicate Records
    # ========================================================

    def duplicate_records(self):

        duplicates = self.df.duplicated().sum()

        logger.info("=" * 60)
        logger.info("DUPLICATE RECORDS")
        logger.info("=" * 60)

        print(f"\nDuplicate Records : {duplicates}")

        return duplicates

    # ========================================================
    # Statistical Summary
    # ========================================================

    def statistical_summary(self):

        logger.info("=" * 60)
        logger.info("STATISTICAL SUMMARY")
        logger.info("=" * 60)

        summary = self.df.describe(include="all")

        print(summary)

        return summary

    # ========================================================
    # Unique Values
    # ========================================================

    def unique_values(self):

        logger.info("=" * 60)
        logger.info("UNIQUE VALUES")
        logger.info("=" * 60)

        unique_dict = {}

        for column in self.df.columns:

            unique_dict[column] = self.df[column].nunique()

            print(
                f"{column:<25}"
                f"Unique Values : {self.df[column].nunique()}"
            )

        return unique_dict

    # ========================================================
    # Dataset Report
    # ========================================================

    def generate_report(self):

        logger.info("Generating Data Analysis Report...")

        report_path = config.REPORT_PATH

        with open(report_path, "w", encoding="utf-8") as report:

            report.write("=" * 70 + "\n")
            report.write("LOAN PREDICTION SYSTEM\n")
            report.write("DATA ANALYSIS REPORT\n")
            report.write("=" * 70 + "\n\n")

            report.write(
                f"Generated On : "
                f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n"
            )

            report.write(
                f"Dataset Shape : {self.df.shape}\n\n"
            )

            report.write("=" * 70 + "\n")
            report.write("COLUMN NAMES\n")
            report.write("=" * 70 + "\n\n")

            for column in self.df.columns:
                report.write(f"{column}\n")

            report.write("\n")

            report.write("=" * 70 + "\n")
            report.write("DATA TYPES\n")
            report.write("=" * 70 + "\n\n")

            report.write(
                self.df.dtypes.to_string()
            )

            report.write("\n\n")

            report.write("=" * 70 + "\n")
            report.write("MISSING VALUES\n")
            report.write("=" * 70 + "\n\n")

            report.write(
                self.df.isnull().sum().to_string()
            )

            report.write("\n\n")

            report.write("=" * 70 + "\n")
            report.write("STATISTICAL SUMMARY\n")
            report.write("=" * 70 + "\n\n")

            report.write(
                self.df.describe(include="all").to_string()
            )

            report.write("\n\n")

            report.write("=" * 70 + "\n")
            report.write("UNIQUE VALUES\n")
            report.write("=" * 70 + "\n\n")

            for column in self.df.columns:

                report.write(
                    f"{column:<25}"
                    f"{self.df[column].nunique()}\n"
                )

            report.write("\n")

            report.write("=" * 70 + "\n")
            report.write("OBSERVATIONS\n")
            report.write("=" * 70 + "\n\n")

            report.write(
                f"Total Records : {self.df.shape[0]}\n"
            )

            report.write(
                f"Total Columns : {self.df.shape[1]}\n"
            )

            report.write(
                f"Duplicate Records : "
                f"{self.df.duplicated().sum()}\n"
            )

            report.write(
                f"Columns with Missing Values : "
                f"{self.df.isnull().sum().astype(bool).sum()}\n"
            )

        logger.info(
            f"Report Saved Successfully :\n{report_path}"
        )

            # ========================================================
    # Generate Graphs
    # ========================================================

    def generate_graphs(self):

        logger.info("=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)

        sns.set_style("whitegrid")

        # ----------------------------------------------------
        # Loan Status Distribution
        # ----------------------------------------------------

        plt.figure(figsize=(8, 6))

        sns.countplot(
            data=self.df,
            x="Loan_Status",
            palette="viridis"
        )

        plt.title("Loan Status Distribution")
        plt.xlabel("Loan Status")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(config.LOAN_STATUS_GRAPH)

        plt.close()

        logger.info("loan_status.png created.")

        # ----------------------------------------------------
        # Gender Distribution
        # ----------------------------------------------------

        plt.figure(figsize=(8, 6))

        sns.countplot(
            data=self.df,
            x="Gender",
            palette="Set2"
        )

        plt.title("Gender Distribution")
        plt.xlabel("Gender")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(config.GENDER_GRAPH)

        plt.close()

        logger.info("gender.png created.")

        # ----------------------------------------------------
        # Education Distribution
        # ----------------------------------------------------

        plt.figure(figsize=(8, 6))

        sns.countplot(
            data=self.df,
            x="Education",
            palette="coolwarm"
        )

        plt.title("Education Distribution")
        plt.xlabel("Education")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(config.EDUCATION_GRAPH)

        plt.close()

        logger.info("education.png created.")

        # ----------------------------------------------------
        # Applicant Income Distribution
        # ----------------------------------------------------

        plt.figure(figsize=(10, 6))

        sns.histplot(
            self.df["ApplicantIncome"],
            bins=30,
            kde=True,
            color="royalblue"
        )

        plt.title("Applicant Income Distribution")
        plt.xlabel("Applicant Income")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(config.INCOME_GRAPH)

        plt.close()

        logger.info("income.png created.")

        # ----------------------------------------------------
        # Correlation Heatmap
        # ----------------------------------------------------

        correlation_df = self.df.copy()

        for column in correlation_df.columns:

            if correlation_df[column].dtype == "object":

                encoder = LabelEncoder()

                correlation_df[column] = encoder.fit_transform(
                    correlation_df[column].astype(str)
                )

        plt.figure(figsize=(12, 8))

        sns.heatmap(
            correlation_df.corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(config.CORRELATION_GRAPH)

        plt.close()

        logger.info("correlation.png created.")

        logger.info("All graphs generated successfully.")

    # ========================================================
    # Clean Dataset
    # ========================================================

    def clean_data(self):

        logger.info("=" * 60)
        logger.info("DATA CLEANING")
        logger.info("=" * 60)

        # -------------------------------
        # Remove Duplicate Records
        # -------------------------------

        duplicate_rows = self.df.duplicated().sum()

        if duplicate_rows > 0:

            self.df.drop_duplicates(inplace=True)

            logger.info(
                f"Removed {duplicate_rows} duplicate rows."
            )

        else:

            logger.info("No duplicate records found.")

        # -------------------------------
        # Numerical Columns
        # -------------------------------

        numerical_columns = [

            "ApplicantIncome",

            "CoapplicantIncome",

            "LoanAmount",

            "Loan_Amount_Term",

            "Credit_History"

        ]

        for column in numerical_columns:

            if self.df[column].isnull().sum() > 0:

                mean_value = self.df[column].mean()

                self.df[column].fillna(

                    mean_value,

                    inplace=True

                )

                logger.info(

                    f"{column} "

                    f"missing values filled using mean."

                )

        # -------------------------------
        # Categorical Columns
        # -------------------------------

        categorical_columns = [

            "Gender",

            "Married",

            "Dependents",

            "Self_Employed"

        ]

        for column in categorical_columns:

            if self.df[column].isnull().sum() > 0:

                mode_value = self.df[column].mode()[0]

                self.df[column].fillna(

                    mode_value,

                    inplace=True

                )

                logger.info(

                    f"{column} "

                    f"missing values filled using mode."

                )

        logger.info("=" * 60)

        logger.info("Missing Values After Cleaning")

        logger.info("=" * 60)

        print(self.df.isnull().sum())

        logger.info("Dataset cleaned successfully.")

            # ========================================================
    # Encode Categorical Features
    # ========================================================

    def encode_features(self):

        logger.info("=" * 60)
        logger.info("LABEL ENCODING")
        logger.info("=" * 60)

        categorical_columns = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        if "Loan_ID" in categorical_columns:
            categorical_columns.remove("Loan_ID")

        for column in categorical_columns:

            encoder = LabelEncoder()

            self.df[column] = encoder.fit_transform(
                self.df[column].astype(str)
            )

            self.label_encoders[column] = encoder

            logger.info(f"{column} encoded successfully.")

        logger.info("All categorical columns encoded.")

    # ========================================================
    # Save Label Encoders
    # ========================================================

    def save_label_encoders(self):

        logger.info("Saving Label Encoders...")

        joblib.dump(

            self.label_encoders,

            config.ENCODER_PATH

        )

        logger.info(
            f"Label encoders saved at:\n{config.ENCODER_PATH}"
        )

    # ========================================================
    # Save Cleaned Dataset
    # ========================================================

    def save_clean_dataset(self):

        logger.info("Saving cleaned dataset...")

        cleaned_df = self.df.copy()

        cleaned_df.to_csv(

            config.ENCODED_DATASET_PATH,

            index=False

        )

        logger.info(
            f"Encoded dataset saved at:\n{config.ENCODED_DATASET_PATH}"
        )

    # ========================================================
    # Complete Pipeline
    # ========================================================

    def run_pipeline(self):

        logger.info("=" * 60)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        self.load_dataset()

        self.dataset_information()

        self.missing_values()

        self.duplicate_records()

        self.statistical_summary()

        self.unique_values()

        self.generate_report()

        self.generate_graphs()

        self.clean_data()

        self.encode_features()

        self.save_label_encoders()

        self.save_clean_dataset()

        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)


# ============================================================
# Main Function
# ============================================================

def main():

    processor = LoanDataPreprocessor()

    processor.run_pipeline()


# ============================================================
# Program Entry
# ============================================================

if __name__ == "__main__":

    main()