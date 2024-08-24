from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el pipeline de preprocesamiento y el modelo
scaler_path = 'model/files/scaler.pkl'
model_path = 'model/files/ml_model.pkl'

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

required_columns = {'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'}

def validate_data(data):
    data_keys = set(data.keys())
    missing_columns = required_columns - data_keys
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")

def make_prediction(data):
    df = pd.DataFrame([data])
    X = scaler.transform(df)
    prediction = model.predict(X)
    return prediction

# Endpoint para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        validate_data(data)
        prediction = make_prediction(data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
