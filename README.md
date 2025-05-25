# Mushroom Classification App

An interactive web application for classifying mushrooms as edible or poisonous based on their features. This application allows users to explore the dataset, choose from various machine learning models, and evaluate their performance.

## Features

*   Upload your own mushroom data (CSV format). *(Note: File uploader functionality to be added in a future step. Currently loads `mushroom.csv` by default.)*
*   View raw and preprocessed dataset.
*   Select from various classification models:
    *   Support Vector Machine (SVM)
    *   Random Forest
    *   AdaBoost
    *   Gradient Boosting
    *   XGBoost
    *   LightGBM
*   Train the selected model and view performance metrics (accuracy, classification report, confusion matrix).

## Setup and Installation

1.  **Python Version:**
    *   Python 3.8+ is recommended.

2.  **Create a Virtual Environment (Optional but Recommended):**
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    *   Dependencies are listed in `requirements.txt`. (Note: This file will be created in a later step).
    *   Install them using pip:
        ```bash
        pip install -r requirements.txt
        ```
    *   Main dependencies include:
        *   streamlit
        *   pandas
        *   scikit-learn
        *   xgboost
        *   lightgbm
        *(Exact versions will be specified in `requirements.txt`)*

## How to Run the Application

1.  Navigate to the project directory in your terminal.
2.  Ensure all dependencies are installed (see Setup and Installation).
3.  Run the Streamlit application using the command:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your default web browser.

## How to Use the Application

*   **Data Loading**:
    *   The application loads the `mushroom.csv` dataset by default upon starting.
*   **Data Viewing**:
    *   You can view the raw uploaded data and the preprocessed data in interactive tables within the app.
*   **Model Selection**:
    *   Choose your preferred classification model from the dropdown menu.
*   **Training and Evaluation**:
    *   Click the "Train and Evaluate Model" button. This will:
        *   Train the selected model on an 80% split of the data.
        *   Evaluate the model on the remaining 20% of the data.
    *   Performance metrics including accuracy, a detailed classification report, and a confusion matrix will be displayed.

## Dataset

The default dataset used is `mushroom.csv`. This dataset contains various physical characteristics of mushrooms, which are used to classify them as either edible or poisonous.

---

*This README provides a guide to setting up and using the Mushroom Classification App.*
