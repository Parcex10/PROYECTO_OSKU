import pathlib
import pickle
from prefect import task, flow
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import dagshub

dagshub.init(repo_owner='colome8', repo_name='PROYECTO_OSKU', mlflow=True)

# Configurar el tracking URI de MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/colome8/PROYECTO_OSKU.mlflow"  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Nombre del experimento
EXPERIMENT_NAME = "prefect-model-experiment"
MODEL_REGISTRY_NAME = "prefect-modelos"

try:
    client.get_registered_model(MODEL_REGISTRY_NAME)
    print(f"El registro de modelos '{MODEL_REGISTRY_NAME}' ya existe.")
except:
    client.create_registered_model(MODEL_REGISTRY_NAME)
    print(f"Registro de modelos '{MODEL_REGISTRY_NAME}' creado.")

mlflow.set_experiment(EXPERIMENT_NAME)

# cargar datos
@task
def read_dataframe(filename):

    df = pd.read_csv(filename)

    categorical = ['label', 'device']
    df[categorical] = df[categorical].astype(str)

    return df


# Preprocess
@task
def preprocess_columns(df: pd.DataFrame):
    df.dropna(inplace=True)  # Eliminar valores faltantes
    return df



# dividir datos
@task
def split_data(df: pd.DataFrame):
    X = df[['sessions', 'drives', 'total_sessions']]  # Seleccionar características
    y = df['label'].apply(lambda x: 1 if x == 'retained' else 0)  # Convertir a variable binaria (1 para retenido, 0 para no retenido)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Guardar el label encoder
    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    return train_test_split(X, y, test_size=0.3, random_state=42)


@task
def train_and_save_scaler(X_train):
    # Entrenar el escalador
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Guardar el escalador
    pathlib.Path("models").mkdir(exist_ok=True, parents=True)
    with open("models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    
    return X_train_scaled   


# Entrenar modelo
@task
def run_models(X_train, X_test, y_train, y_test):
    models = [
        {"model": LogisticRegression, "params": {}},
        {"model": DecisionTreeClassifier, "params": {}},
        {"model": RandomForestClassifier, "params": {}},
        {"model": SVC, "params": {"probability": True}},
    ]

    # Iniciar un run de MLflow para los modelos anidados
    with mlflow.start_run(run_name="Prefect-Nested-Runs"):
        for model in models:
            model_class = model["model"]
            model_name = model_class.__name__
            params = model["params"]

            with mlflow.start_run(run_name="Prefect-" + model_name, nested=True):
                # Loguear parámetros
                for param, value in params.items():
                    mlflow.log_param(param, value)

                # Entrenar el modelo
                ml_model = model_class(**params)
                ml_model.fit(X_train, y_train)

                # Predicciones
                y_pred = ml_model.predict(X_test)

                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy", accuracy)
                
                label_encoder_path = "C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\PROYECTO_OSKU\\models\\label_encoder.pkl"
        
                if pathlib.Path(label_encoder_path).exists():
                    mlflow.log_artifact(label_encoder_path, artifact_path="artifacts")
                else:
                    print(f"Advertencia: {label_encoder_path} no encontrado.")

                scaler_path = "C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\PROYECTO_OSKU\\models\\scaler.pkl"
        
                if pathlib.Path(scaler_path).exists():
                    mlflow.log_artifact(scaler_path, artifact_path="artifacts")
                else:
                    print(f"Advertencia: {scaler_path} no encontrado.")    

                mlflow.sklearn.log_model(ml_model, "modelos")

                print(f"Modelo '{model_name}' registrado con precisión: {accuracy:.4f}")


@task
def randomized_search(X_train, X_test, y_train, y_test):
    # Modelos y parámetros
    models = [
        {
            "model": LogisticRegression,
            "params": {"C": [0.1, 1.0, 10], "solver": ["liblinear", "lbfgs"]},
        },
        {
            "model": DecisionTreeClassifier,
            "params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]},
        },
        {
            "model": RandomForestClassifier,
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
        },
    ]

    # Ejecutar runs anidados
    with mlflow.start_run(run_name="Prefect-Hyperparameter-Tuning"):
        for model in models:
            model_class = model["model"]
            model_name = model_class.__name__
            param_grid = model["params"]

            with mlflow.start_run(run_name="Prefect-Tuned-" + model_name, nested=True):
                # Loguear el modelo y su grid de hiperparámetros
                mlflow.log_param("param_grid", param_grid)

                # Configurar Randomized search
                randomized_search = RandomizedSearchCV(
                estimator=model_class(),
                param_distributions=param_grid,
                n_iter=3,  
                scoring="accuracy",
                cv=2,
                n_jobs=-1
                )


                # Ajustar modelo
                start_time = time.time()
                randomized_search.fit(X_train, y_train)
                duration = time.time() - start_time

                # Mejor modelo y resultados
                best_model = randomized_search.best_estimator_
                best_params = randomized_search.best_params_
                best_score = randomized_search.best_score_

                # Predicciones en conjunto de prueba
                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Loguear métricas y parámetros
                mlflow.log_param("best_params", best_params)
                mlflow.log_metric("cv_accuracy", best_score)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("tuning_duartion", duration)

                # loguear label encoder
                label_encoder_path = "C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\PROYECTO_OSKU\\models\\label_encoder.pkl"
        
                if pathlib.Path(label_encoder_path).exists():
                    mlflow.log_artifact(label_encoder_path, artifact_path="artifacts")
                else:
                    print(f"Advertencia: {label_encoder_path} no encontrado.")

                scaler_path = "C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\PROYECTO_OSKU\\models\\scaler.pkl"
        
                if pathlib.Path(scaler_path).exists():
                    mlflow.log_artifact(scaler_path, artifact_path="artifacts")
                else:
                    print(f"Advertencia: {scaler_path} no encontrado.")

                mlflow.sklearn.log_model(best_model, "modelos")

                print(f"Modelo '{model_name}' registrado con precisión en prueba: {accuracy:.4f}")




# Log y registro de modelos en MLflow
@task
def model_registry_and_alias(model_registry_name, experiment_name):
    client = MlflowClient()

    # Obtener el ID del experimento
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    # Buscar y ordenar las runs por accuracy
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        order_by=["metrics.accuracy DESC"],  # Ordenar por mayor accuracy
        max_results=10  # Opcional: limitar el número de runs
    )

    try:
        client.get_registered_model(model_registry_name)
        print(f"El registro de modelos '{model_registry_name}' ya existe.")
    except:
        client.create_registered_model(model_registry_name)
        print(f"Registro de modelos '{model_registry_name}' creado.")

    # Asignar Champion y Challenger
    if len(runs) >= 2:
        # Run con mayor accuracy
        best_run = runs[0]
        second_best_run = runs[1]

        # Registrar modelos
        best_model_version = client.create_model_version(
            name=model_registry_name,
            source=f"runs:/{best_run.info.run_id}/model",  # Ruta del modelo en la run
            run_id=best_run.info.run_id
        )

        second_best_model_version = client.create_model_version(
            name=model_registry_name,
            source=f"runs:/{second_best_run.info.run_id}/model",
            run_id=second_best_run.info.run_id
        )

        # Asignar Champion
        client.transition_model_version_stage(
            name=model_registry_name,
            version=best_model_version.version,
            stage="Production"
        )
        client.set_registered_model_alias(model_registry_name, "Champion", best_model_version.version)

        # Asignar Challenger
        client.transition_model_version_stage(
            name=model_registry_name,
            version=second_best_model_version.version,
            stage="Staging"
        )
        client.set_registered_model_alias(model_registry_name, "Challenger", second_best_model_version.version)

        print(f"Champion: Run ID {best_run.info.run_id}, Accuracy: {best_run.data.metrics['accuracy']}")
        print(f"Challenger: Run ID {second_best_run.info.run_id}, Accuracy: {second_best_run.data.metrics['accuracy']}")
    else:
        print("No hay suficientes runs para asignar Champion y Challenger.")


# Orquestación las tareas
@flow
def project_flow(path):
    # 1. Cargar datos crudos
    df = read_dataframe(path)
    
    # 2. Preprocesar datos
    df = preprocess_columns(df)
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Save dict vectorizer
    train_and_save_scaler(X_train)
    
    # 4. Entrenar y loguear
    run_models(X_train, X_test, y_train, y_test)
    randomized_search(X_train, X_test, y_train, y_test)

    # model registry and alias assign
    model_registry_and_alias(MODEL_REGISTRY_NAME, EXPERIMENT_NAME)


# Ejecutar el pipeline
if __name__ == '__main__':
    project_flow("C:\\Users\\colom\\OneDrive - ITESO\\iteso\\5to semestre\\cienciadatos\\PROYECTO_OSKU\\data\\waze_dataset.csv")

