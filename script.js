/*
==========================================================
Loan Prediction System
Client Side JavaScript
==========================================================
*/

"use strict";

// ==========================================================
// DOM Loaded
// ==========================================================

document.addEventListener("DOMContentLoaded", function () {

    console.log("Loan Prediction System Loaded Successfully.");

    const form = document.getElementById("loanForm");
    const submitButton = document.getElementById("predictBtn");

    if (!form) {
        return;
    }

    // ======================================================
    // Prevent Negative Numbers
    // ======================================================

    const numericFields = document.querySelectorAll(
        "input[type='number']"
    );

    numericFields.forEach(function (field) {

        field.addEventListener("input", function () {

            if (this.value < 0) {

                this.value = "";

            }

        });

    });

    // ======================================================
    // Bootstrap Validation
    // ======================================================

    form.addEventListener("submit", function (event) {

        if (!form.checkValidity()) {

            event.preventDefault();

            event.stopPropagation();

            form.classList.add("was-validated");

            return;

        }

        // ==============================================
        // Disable Button
        // ==============================================

        submitButton.disabled = true;

        submitButton.innerHTML =

            '<span class="spinner-border spinner-border-sm me-2"></span>' +

            'Predicting...';

    });

});

// ==========================================================
// Utility Functions
// ==========================================================

function onlyPositive(element) {

    if (element.value < 0) {

        element.value = "";

    }

}

function onlyInteger(element) {

    element.value = parseInt(element.value) || "";

}

// ==========================================================
// Auto Focus First Field
// ==========================================================

window.onload = function () {

    const firstField = document.querySelector(

        "select, input"

    );

    if (firstField) {

        firstField.focus();

    }

};

// ==========================================================
// Console Banner
// ==========================================================

console.log(
    "%cLoan Prediction System Ready",
    "color:green;font-size:16px;font-weight:bold;"
);

console.log(
    "Machine Learning Model : Random Forest"
);

console.log(
    "Frontend Loaded Successfully."
);