import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Set title for the Streamlit application
st.title("Mushroom Classification App")

# Function to load data
def load_data():
    """Loads the mushroom dataset from a CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv("mushroom.csv")
        return df
    except FileNotFoundError:
        st.error("Error: mushroom.csv not found. Please make sure the file is in the correct directory.")
        return None

# Main part of the app
if __name__ == "__main__":
    df = load_data()

    if df is not None:
        st.subheader("Mushroom Dataset")
        st.dataframe(df)

        def preprocess_data(df_to_process):
            """Preprocesses the mushroom dataset.

            Args:
                df_to_process (pandas.DataFrame): The input DataFrame.

            Returns:
                pandas.DataFrame: The preprocessed DataFrame.
            """
            # Drop "Unnamed: 0" column if it exists
            if "Unnamed: 0" in df_to_process.columns:
                df_to_process = df_to_process.drop("Unnamed: 0", axis=1)

            # Separate categorical and numerical features (simplified for this dataset)
            # For mushroom.csv, all descriptive columns are categorical.
            # 'class' is the target variable.

            label_encoder = LabelEncoder()

            for col in df_to_process.columns:
                if df_to_process[col].dtype == 'object':
                    df_to_process[col] = label_encoder.fit_transform(df_to_process[col])
            
            # Encode target variable 'class'
            if 'class' in df_to_process.columns:
                 df_to_process['class'] = label_encoder.fit_transform(df_to_process['class'])


            return df_to_process

        processed_df = preprocess_data(df.copy())

        st.subheader("Preprocessed Data")
        st.dataframe(processed_df)

        # Define features X and target y
        X = processed_df.drop('class', axis=1)
        y = processed_df['class']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        model_choice = st.selectbox(
            "Choose a model:",
            ("SVM", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM")
        )

        # Instantiate model based on choice
        if model_choice == "SVM":
            model = SVC()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "AdaBoost":
            model = AdaBoostClassifier(random_state=42)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif model_choice == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_choice == "LightGBM":
            model = lgb.LGBMClassifier(random_state=42)

        # Train and Evaluate Model button
        if st.button("Train and Evaluate Model"):
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.4f}")

            # Display classification report
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Display confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.dataframe(cm)
