import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Cargar el pipeline de preprocesamiento
def load_preprocessing_pipeline():
    pipeline_path = '../files/data_pipeline.pkl'
    return joblib.load(pipeline_path)

# Cargar y procesar el dataset
#def load_and_process_data(filepath, preprocessor):
def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                  'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 
                  'max_heart_rate_achieved', 'exercise_induced_angina', 
                  'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia',
                  'target']
    df = df.dropna()
    
    X = df.drop(columns='target')
    y = df['target']
    
    # Aplicar el pipeline de preprocesamiento
    #X_processed = preprocessor.transform(X)
    
    #return X_processed, y
    return X, y

# Separar datos en conjuntos de entrenamiento y prueba
def separate_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Entrenar y ajustar modelos
def train_models(X_train, y_train, X_test, y_test):
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6),
        'Random Forest': RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5),
        'Support Vector Machine': SVC(kernel='rbf', C=2),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=10)
    }

    param_grids = {
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly']
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }

    results = {}

    for name, clf in classifiers.items():
        grid_search = GridSearchCV(clf, param_grids[name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_clf = grid_search.best_estimator_

        y_pred = best_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        results[name] = {
            'Best Estimator': best_clf,
            'Accuracy': accuracy,
            'Classification Report': report
        }

    return results

# Evaluar y visualizar los resultados de los modelos
def evaluate_models(results):
    classifier_names = list(results.keys())
    accuracies = [results[name]['Accuracy'] for name in classifier_names]

    plt.figure(figsize=(10, 5))
    plt.bar(classifier_names, accuracies, color='skyblue')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.ylim([0, 1])
    for i in range(len(classifier_names)):
        plt.text(i, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', ha='center')
    plt.show()

# Guardar el mejor modelo
def save_best_model(results):
    best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
    best_model = results[best_model_name]['Best Estimator']
    directory = os.path.join(os.path.dirname(__file__), "..", "files")
    model_path = os.path.join(directory, 'ml_model.pkl')
    joblib.dump(best_model, model_path)
    return model_path

# Cargar el mejor modelo
def load_model(model_filepath):
    return joblib.load(model_filepath)

# Ejecutar el pipeline de entrenamiento y evaluación
def run_ml(filepath):
    #preprocessor = load_preprocessing_pipeline()
    #X, y = load_and_process_data(filepath, preprocessor)
    X, y = load_and_process_data(filepath)
    X_train, X_test, y_train, y_test = separate_data(X, y)
    results = train_models(X_train, y_train, X_test, y_test)
    evaluate_models(results)
    save_best_model(results)

# Ejecutar el pipeline
if __name__ == "__main__":
    filepath = '../model/data/heart.csv'
    run_ml(filepath)