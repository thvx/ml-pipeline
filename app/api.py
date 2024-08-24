from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el pipeline de preprocesamiento y el modelo
pipeline_path = 'model/files/data_pipeline.pkl'
model_path = 'model/files/ml_model.pkl'

pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)

def validate_data(data):
    required_columns = {'age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression',
                        'chest_pain_type', 'rest_ecg', 'st_slope', 'num_major_vessels', 'thalassemia'}
    missing_columns = required_columns - data.keys()
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")

def predict(data):
    df = pd.DataFrame([data])
    X = pipeline.transform(df)
    prediction = model.predict(X)
    return prediction

# Endpoint para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        validate_data(data)
        prediction = predict(data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
