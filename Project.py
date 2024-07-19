import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("heart.csv")

# Preprocessing
categorical_data = []
continuous_data = []
for col in data.columns:
    if len(data[col].unique()) <= 10:
        categorical_data.append(col)
    else:
        continuous_data.append(col)

# Remove target from categorical data
categorical_data.remove('target')

# One-hot encode categorical columns
dataset = pd.get_dummies(data, columns=categorical_data)

# Standardize continuous columns
scaler = StandardScaler()
continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[continuous_columns] = scaler.fit_transform(dataset[continuous_columns])

# Split the data into features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Train the logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=92)
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

def get_user_input():
    # Define input fields
    user_data = {
        'age': st.number_input('Age', min_value=0, max_value=120, value=25),
        'trestbps': st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120),
        'chol': st.number_input('Cholesterol', min_value=0, max_value=600, value=200),
        'thalach': st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, value=130),
        'oldpeak': st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0),
        'sex': st.selectbox('Sex', options=['Male', 'Female']),
        'cp': st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']),
        'restecg': st.selectbox('Resting ECG', options=['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']),
        'exang': st.selectbox('Exercise Induced Angina', options=['No', 'Yes']),
        'thal': st.selectbox('Thalassemia', options=['Normal', 'Fixed Defect', 'Reversible Defect'])
    }

    # Convert categorical inputs to one-hot encoding
    user_data_encoded = {
        'age': user_data['age'],
        'trestbps': user_data['trestbps'],
        'chol': user_data['chol'],
        'thalach': user_data['thalach'],
        'oldpeak': user_data['oldpeak'],
        'sex_0': int(user_data['sex'] == 'Female'),
        'sex_1': int(user_data['sex'] == 'Male'),
        'cp_0': int(user_data['cp'] == 'Typical Angina'),
        'cp_1': int(user_data['cp'] == 'Atypical Angina'),
        'cp_2': int(user_data['cp'] == 'Non-Anginal Pain'),
        'cp_3': int(user_data['cp'] == 'Asymptomatic'),
        'restecg_0': int(user_data['restecg'] == 'Normal'),
        'restecg_1': int(user_data['restecg'] == 'ST-T Wave Abnormality'),
        'restecg_2': int(user_data['restecg'] == 'Left Ventricular Hypertrophy'),
        'exang_0': int(user_data['exang'] == 'No'),
        'exang_1': int(user_data['exang'] == 'Yes'),
        'thal_1': int(user_data['thal'] == 'Normal'),
        'thal_2': int(user_data['thal'] == 'Fixed Defect'),
        'thal_3': int(user_data['thal'] == 'Reversible Defect')
    }

    # Create DataFrame for user inputs
    user_input_df = pd.DataFrame([user_data_encoded])

    # Ensure all columns are present
    for col in X.columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0

    # Reorder columns to match the training data
    user_input_df = user_input_df[X.columns]

    return user_input_df

# Streamlit app
st.title('Heart Disease Prediction')

# Get user input
user_input = get_user_input()

# Add a button for prediction
if st.button('Predict'):
    # Standardize the user input for continuous columns
    user_input[continuous_columns] = scaler.transform(user_input[continuous_columns])

    # Make predictions
    prediction = lr_clf.predict(user_input)
    prediction_proba = lr_clf.predict_proba(user_input)

    # Display the results
    st.write('Prediction: ', 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
    st.write('Prediction Probability: ', prediction_proba[0])

    # Generate and display a simulated ECG graph
    st.write('Simulated ECG Graph:')
    time = np.linspace(0, 1, 500)
    ecg_signal = np.sin(5 * 2 * np.pi * time) + np.random.normal(0, 0.1, time.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(time, ecg_signal)
    plt.title('Simulated ECG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    st.pyplot(plt)
