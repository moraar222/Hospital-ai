# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
# Assuming a dataset with columns: 'age', 'gender', 'symptoms', 'health_history', 'disease', 'specialist'
data = pd.read_csv('patient_data.csv')

# Preprocess the data
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['disease'] = label_encoder.fit_transform(data['disease'])
data['symptoms'] = label_encoder.fit_transform(data['symptoms'])
data['health_history'] = label_encoder.fit_transform(data['health_history'])

# Split the data into training and testing sets
X = data[['age', 'gender', 'symptoms', 'health_history']]
y = data['disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'disease_prediction_model.pkl')

# Function to predict the disease and recommend a specialist
def predict_disease_and_recommend_specialist(age, gender, symptoms, health_history):
    # Load the model
    model = joblib.load('disease_prediction_model.pkl')
    
    # Preprocess input data
    gender_encoded = label_encoder.transform([gender])[0]
    symptoms_encoded = label_encoder.transform([symptoms])[0]
    health_history_encoded = label_encoder.transform([health_history])[0]
    
    # Predict the disease
    disease = model.predict([[age, gender_encoded, symptoms_encoded, health_history_encoded]])[0]
    disease_name = label_encoder.inverse_transform([disease])[0]
    
    # Recommend the specialist
    specialist = data[data['disease'] == disease]['specialist'].values[0]
    
    return disease_name, specialist

# Example usage
if __name__ == "__main__":
    age = 45
    gender = 'male'
    symptoms = 'chest pain'
    health_history = 'heart disease'
    
    disease, specialist = predict_disease_and_recommend_specialist(age, gender, symptoms, health_history)
    
    print(f"Predicted Disease: {disease}")
    print(f"Recommended Specialist: {specialist}")
