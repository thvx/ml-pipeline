from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report

app = Flask(__name__)

# Cargar el pipeline de preprocesamiento y el modelo
pipeline = joblib.load('model/files/data_pipeline.pkl')
model = joblib.load('model/files/ml_model.pkl')

# Endpoint para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    X = pipeline.transform(df)
    
    prediction = model.predict(X)
    return jsonify({'prediction': prediction[0]})

# Endpoint para extraer una fila del CSV
@app.route('/extract_row', methods=['GET'])
def extract_row():
    row_number = int(request.args.get('row_number'))
    df = pd.read_csv('model/data/heart.csv')
    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                  'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 
                  'max_heart_rate_achieved','exercise_induced_angina', 
                  'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia',
                  'target']
    df = df.dropna()
    row_data = df.iloc[row_number].to_dict()
    
    return jsonify(row_data)

# Endpoint para predecir una fila específica
@app.route('/predict_row', methods=['GET'])
def predict_row():
    row_number = int(request.args.get('row_number'))
    df = pd.read_csv('model/data/heart.csv')
    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                  'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 
                  'max_heart_rate_achieved','exercise_induced_angina', 
                  'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia',
                  'target']
    df = df.dropna()
    
    row_data = df.iloc[row_number]
    X = pipeline.transform(pd.DataFrame([row_data]))
    
    prediction = model.predict(X)
    return jsonify({'prediction': prediction[0]})

# Endpoint para evaluar todos los datos en test.json
@app.route('/evaluate_all', methods=['GET'])
def evaluate_all():
    with open('test.json') as f:
        test_data = json.load(f)
    
    test_df = pd.DataFrame(test_data)
    X_test = pipeline.transform(test_df)
    y_test = test_df['target']
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)
