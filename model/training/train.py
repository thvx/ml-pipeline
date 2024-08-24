import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from matplotlib.pylab import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plot style
rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.facecolor'] = 'white'

# Preprocesamiento
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Exportación
import joblib

# Entrenamiento de modelos
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Funciones de Preprocesamiento y Feature Engineering
def array_to_df(X, columns):
    return pd.DataFrame(X, columns=columns)

def preprocessor_pipeline():
    numeric_features = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
    categorical_features = ['chest_pain_type', 'rest_ecg', 'st_slope', 'num_major_vessels', 'thalassemia']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_and_save_pipeline():
    preprocessor = preprocessor_pipeline()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    file_path = '../files/data_pipeline.pkl'
    ensure_dir(file_path)
    joblib.dump(pipeline, file_path)

# Cargar el pipeline de preprocesamiento
def load_preprocessing_pipeline():
    pipeline_path = '../files/data_pipeline.pkl'
    return joblib.load(pipeline_path)

# Cargar y procesar el dataset
def load_and_process_data(filepath, preprocessor):
    df = pd.read_csv(filepath)
    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                  'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 
                  'max_heart_rate_achieved', 'exercise_induced_angina', 
                  'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia',
                  'target']
    df = df.dropna()

    X = df.drop(columns='target')
    y = df['target']
    
    # Ensure that columns exist before applying transformations
    required_columns = set(['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression'])
    if not required_columns.issubset(X.columns):
        raise ValueError(f"Missing columns in input data: {required_columns - set(X.columns)}")

    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

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
    create_and_save_pipeline()
    preprocessor = load_preprocessing_pipeline()
    X, y = load_and_process_data(filepath, preprocessor)
    X_train, X_test, y_train, y_test = separate_data(X, y)
    results = train_models(X_train, y_train, X_test, y_test)
    evaluate_models(results)
    save_best_model(results)

# Ejecutar el pipeline
if __name__ == "__main__":
    filepath = 'model/training/heart.csv'
    run_ml(filepath)
