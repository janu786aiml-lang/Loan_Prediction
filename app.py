"""
=============================================================
Loan Prediction System
Flask Web Application

Author : Your Name
=============================================================
"""

# ==========================================================
# Standard Library Imports
# ==========================================================

import logging
import warnings

# ==========================================================
# Third Party Imports
# ==========================================================

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request
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
# Flask App
# ==========================================================

app = Flask(__name__)

# ==========================================================
# Load Saved Objects
# ==========================================================

logger.info("=" * 60)
logger.info("LOADING TRAINED MODEL")
logger.info("=" * 60)

try:

    model = joblib.load(
        config.MODEL_PATH
    )

    scaler = joblib.load(
        config.SCALER_PATH
    )

    label_encoders = joblib.load(
        config.LABEL_ENCODERS_PATH
    )

    logger.info("Model Loaded Successfully.")

    logger.info("Scaler Loaded Successfully.")

    logger.info("Label Encoders Loaded Successfully.")

except Exception as e:

    logger.error(str(e))

    raise

# ==========================================================
# Mapping Dictionaries
# ==========================================================

GENDER = {

    "Male": 1,

    "Female": 0

}

MARRIED = {

    "Yes": 1,

    "No": 0

}

DEPENDENTS = {

    "0": 0,

    "1": 1,

    "2": 2,

    "3+": 3

}

EDUCATION = {

    "Graduate": 0,

    "Not Graduate": 1

}

SELF_EMPLOYED = {

    "No": 0,

    "Yes": 1

}

PROPERTY_AREA = {

    "Rural": 0,

    "Semiurban": 1,

    "Urban": 2

}

# ==========================================================
# Numerical Columns
# ==========================================================

NUMERICAL_COLUMNS = [

    "ApplicantIncome",

    "CoapplicantIncome",

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History"

]

# ==========================================================
# Model Feature Order
# ==========================================================

MODEL_COLUMNS = [

    "Gender",

    "Married",

    "Dependents",

    "Education",

    "Self_Employed",

    "ApplicantIncome",

    "CoapplicantIncome",

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History",

    "Property_Area"

]

# ==========================================================
# Home Page
# ==========================================================

@app.route("/")

def home():

    logger.info("Home Page Opened.")

    return render_template(

        "index.html"

    )
# ==========================================================
# Prediction Route
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        logger.info("=" * 60)
        logger.info("NEW PREDICTION REQUEST")
        logger.info("=" * 60)

        # --------------------------------------------------
        # Read Form Values
        # --------------------------------------------------

        gender = request.form["gender"]

        married = request.form["married"]

        dependents = request.form["dependents"]

        education = request.form["education"]

        self_employed = request.form["self_employed"]

        applicant_income = float(
            request.form["applicant_income"]
        )

        coapplicant_income = float(
            request.form["coapplicant_income"]
        )

        loan_amount = float(
            request.form["loan_amount"]
        )

        loan_term = float(
            request.form["loan_amount_term"]
        )

        credit_history = float(
            request.form["credit_history"]
        )

        property_area = request.form["property_area"]

        # --------------------------------------------------
        # Convert Categorical Values
        # --------------------------------------------------

        gender = GENDER[gender]

        married = MARRIED[married]

        dependents = DEPENDENTS[dependents]

        education = EDUCATION[education]

        self_employed = SELF_EMPLOYED[self_employed]

        property_area = PROPERTY_AREA[property_area]

        # --------------------------------------------------
        # Create Input DataFrame
        # --------------------------------------------------

        input_df = pd.DataFrame([{

            "Gender": gender,

            "Married": married,

            "Dependents": dependents,

            "Education": education,

            "Self_Employed": self_employed,

            "ApplicantIncome": applicant_income,

            "CoapplicantIncome": coapplicant_income,

            "LoanAmount": loan_amount,

            "Loan_Amount_Term": loan_term,

            "Credit_History": credit_history,

            "Property_Area": property_area

        }])

        # --------------------------------------------------
        # Ensure Correct Column Order
        # --------------------------------------------------

        input_df = input_df[MODEL_COLUMNS]

        # --------------------------------------------------
        # Scale Numerical Columns
        # --------------------------------------------------

        input_df[NUMERICAL_COLUMNS] = scaler.transform(

            input_df[NUMERICAL_COLUMNS]

        )

        logger.info("Input Data Prepared Successfully.")

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        confidence = round(

            max(probability) * 100,

            2

        )

        # --------------------------------------------------
        # Prediction Result
        # --------------------------------------------------

        if prediction == 1:

            result = "Loan Approved"

            message = (
                "Congratulations! "
                "The applicant is eligible for the loan."
            )

        else:

            result = "Loan Rejected"

            message = (
                "Sorry! "
                "The applicant is not eligible for the loan."
            )

        logger.info(f"Prediction : {result}")

        logger.info(f"Confidence : {confidence}%")

        return render_template(

            "result.html",

            prediction=result,

            confidence=confidence,

            message=message

        )

    except Exception as e:

        logger.exception("Prediction Error")

        return render_template(

            "result.html",

            prediction="Error",

            confidence=0,

            message=str(e)

        )
# ==========================================================
# Health Check Route
# ==========================================================

@app.route("/health")
def health():

    logger.info("Health Check Requested.")

    return {

        "status": "success",

        "application": "Loan Prediction System",

        "model_loaded": model is not None,

        "scaler_loaded": scaler is not None

    }

# ==========================================================
# About Route (Optional)
# ==========================================================

@app.route("/about")
def about():

    return {

        "Project": "Loan Prediction System",

        "Framework": "Flask",

        "Machine Learning Model": "Random Forest",

        "Accuracy": "86.18%"

    }
# ==========================================================
# Application Configuration
# ==========================================================

app.config["SECRET_KEY"] = "loan_prediction_system_secret_key"

app.config["TEMPLATES_AUTO_RELOAD"] = True

# ==========================================================
# Startup Banner
# ==========================================================

def startup_banner():

    print("\n")

    print("=" * 70)

    print("            LOAN PREDICTION SYSTEM")

    print("=" * 70)

    print(" Flask Application Started Successfully")

    print(" Best Model      : Random Forest")

    print(" Model Accuracy  : 86.18%")

    print(" URL             : http://127.0.0.1:5000")

    print("=" * 70)

    print("\n")

# ==========================================================
# Main Function
# ==========================================================

def main():

    logger.info("=" * 60)

    logger.info("STARTING FLASK APPLICATION")

    logger.info("=" * 60)

    startup_banner()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True,

        threaded=True

    )

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()