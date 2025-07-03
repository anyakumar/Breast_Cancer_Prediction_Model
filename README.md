# Breast_Cancer_Prediction_Model
A clean and functional breast cancer prediction app built using Streamlit and machine learning. It supports both Logistic Regression and Random Forest models to classify tumors as benign or malignant based on diagnostic features. The project is organized for readability and future updates.

Project Structure
cancer-prediction-model/
│
├── app.py # Streamlit web app
├── data/
│ └── Cancer_Data.csv # Dataset used for training and prediction
├── notebook/
│ └── Breast_Cancer_Prediction_test.ipynb # Jupyter notebook (EDA + model building)
├── model/ # (Optional) Folder for saved models
├── requirements.txt # Python dependencies


---

## ⚙️ How to Run

Make sure Python is installed (preferably 3.10 or 3.11). Then:

```bash
# Clone the repo
git clone https://github.com/yourusername/breast-cancer-prediction-app.git
cd breast-cancer-prediction-app

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


Built-in EDA visualization
Model evaluation metrics
Interactive prediction form
Toggle between Logistic Regression and Random Forest
Organized for clarity and future updates
