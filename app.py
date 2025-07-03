import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.markdown(
    """
    <style>
        /* Main background */
        .stApp {
            background-color: #DDE6D5;
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #EAF4E2;
        }

        /* Sidebar text */
        .css-1d391kg {
            color: #333;
        }

        /* Input box and number input */
        .stNumberInput, .stTextInput {
            background-color: #ffffff;
        }

        /* Titles */
        .stTitle {
            color: #2E4F2E;
        }

        /* Button styling */
        .stButton>button {
            background-color: #6B8E23;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #5A7C1D;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and clean the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/Cancer_Data.csv')
    df = df.drop(['id'], axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)
    df = df.drop_duplicates()
    return df

# Load data
cancer_data = load_data()

# App title
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title(" Breast Cancer Prediction App")
st.write("This app uses Machine Learning to predict whether a tumor is **Malignant or Benign**.")

# Sidebar for model selection
with st.sidebar:
    st.title(" Choose Your Model")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
    st.markdown("---")

# Data visualization
st.subheader(" Distribution of Tumor Types")
sns.countplot(x='diagnosis', data=cancer_data)
st.pyplot(plt.gcf())
plt.clf()

# Encode diagnosis column
cancer_data['diagnosis'] = cancer_data['diagnosis'].map({'M': 1, 'B': 0})

# Prepare data for ML
X = cancer_data.drop('diagnosis', axis=1)
y = cancer_data['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation
y_pred = log_model.predict(X_test)

st.subheader(" Model Evaluation (Logistic Regression)")
st.write("**Accuracy**:", accuracy_score(y_test, y_pred))
st.write("**Confusion Matrix**:")
st.dataframe(confusion_matrix(y_test, y_pred))
st.write("**Classification Report**:")
st.text(classification_report(y_test, y_pred))

# Input features (simplified)
st.subheader(" Enter Tumor Measurements")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input("Mean Radius", min_value=0.0, value=14.0)
    mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, value=90.0)
    mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, value=0.1)

with col2:
    mean_texture = st.number_input("Mean Texture", min_value=0.0, value=20.0)
    mean_area = st.number_input("Mean Area", min_value=0.0, value=600.0)

# Predict button
if st.button(" Predict"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    input_scaled = scaler.transform(input_data)

    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_scaled)
    else:
        prediction = rf_model.predict(input_scaled)

    result = "Malignant" if prediction[0] == 1 else "Benign"

    if result == "Malignant":
        st.error(f"🔴 The tumor is predicted to be **{result}**.")
    else:
        st.success(f"🟢 The tumor is predicted to be **{result}**.")

    st.info(f"Model Used: {model_choice}")

# Footer
st.markdown("---")
st.markdown("🔬 **Breast Cancer Predictor** | Built with ❤️ by *Anya Kumar*")
