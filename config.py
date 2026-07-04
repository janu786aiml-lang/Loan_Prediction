"""
=========================================================
Loan Prediction System
Configuration File
=========================================================
This file stores all project paths and constants.

Author : Your Name
Project: Loan Prediction System
=========================================================
"""

import os

# =========================================================
# Base Directory
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Dataset
# =========================================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "loan_data.csv")

# =========================================================
# Models
# =========================================================

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "loan_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
LABEL_ENCODERS_PATH = os.path.join(
    MODELS_DIR,
    "label_encoders.pkl"
)
# =========================================================
# Encoded Dataset
# =========================================================

ENCODED_DATASET_PATH = os.path.join(DATASET_DIR, "encoded_loan_data.csv")
# =========================================================
# Graphs
# =========================================================

GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")

LOAN_STATUS_GRAPH = os.path.join(GRAPHS_DIR, "loan_status.png")
GENDER_GRAPH = os.path.join(GRAPHS_DIR, "gender.png")
EDUCATION_GRAPH = os.path.join(GRAPHS_DIR, "education.png")
INCOME_GRAPH = os.path.join(GRAPHS_DIR, "income.png")
CORRELATION_GRAPH = os.path.join(GRAPHS_DIR, "correlation.png")

# =========================================================
# Reports
# =========================================================

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORT_PATH = os.path.join(REPORTS_DIR, "data_analysis.txt")
# =========================================================
# Cleaned Dataset
# =========================================================

CLEAN_DATASET_PATH = os.path.join(DATASET_DIR, "cleaned_loan_data.csv")

# =========================================================
# Random State
# =========================================================

RANDOM_STATE = 42

# =========================================================
# Train/Test Split
# =========================================================

TEST_SIZE = 0.20

# =========================================================
# Ensure Required Directories Exist
# =========================================================

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


