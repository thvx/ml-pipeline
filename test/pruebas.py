import unittest
import pandas as pd
import joblib
import json
from flask import Flask
from app import app, pipeline, model  # Asegúrate de que estas importaciones sean correctas

class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.app = app
        cls.app.testing = True

        # Datos de prueba
        cls.sample_data = {
            'age': 63,
            'sex': 1,
            'chest_pain_type': 3,
            'resting_blood_pressure': 145,
            'cholesterol': 233,
            'fasting_blood_sugar': 1,
            'rest_ecg': 0,
            'max_heart_rate_achieved': 150,
            'exercise_induced_angina': 0,
            'st_depression': 2.3,
            'st_slope': 0,
            'num_major_vessels': 0,
            'thalassemia': 1
        }

        cls.row_number = 0

        # Cargar modelo y pipeline para pruebas
        cls.pipeline = pipeline
        cls.model = model

    def test_predict(self):
        response = self.client.post('/predict', json=self.sample_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

    def test_extract_row(self):
        response = self.client.get('/extract_row', query_string={'row_number': self.row_number})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(type(response.json), dict)

    def test_predict_row(self):
        response = self.client.get('/predict_row', query_string={'row_number': self.row_number})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

    def test_evaluate_all(self):
        test_data = [{'age': 50, 'sex': 1, 'chest_pain_type': 4, 'resting_blood_pressure': 140, 'cholesterol': 210, 
                      'fasting_blood_sugar': 0, 'rest_ecg': 1, 'max_heart_rate_achieved': 130, 'exercise_induced_angina': 1, 
                      'st_depression': 1.2, 'st_slope': 2, 'num_major_vessels': 1, 'thalassemia': 0, 'target': 1}]
        with open('test/test.json', 'w') as f:
            json.dump(test_data, f)

        response = self.client.get('/evaluate_all')
        self.assertEqual(response.status_code, 200)
        self.assertIn('target', response.json)

if __name__ == '__main__':
    unittest.main()