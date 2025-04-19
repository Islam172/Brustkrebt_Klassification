import os
import sys

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.utils.predictor import Model
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models, get_classification_score


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


import mlflow
#from urllib.parse import urlparse

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            """
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                
                "Logistic Regression": LogisticRegression(max_iter=1000),
                
            }

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64 ,128],
                },
                
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                },
                "Logistic Regression": {},
            }
             """
            models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42),
}
            params = {
    "Random Forest": {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    "Logistic Regression": {
        'C': [0.1, 1.0, 10.0],  # Regularization strength
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # liblinear supports both l1 and l2
    },
}

            # Evaluate models and find the best one
            model_report: dict = evaluate_models(X_train, y_train, x_test, y_test, models=models, param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
           
            # Log metrics for train and test datasets and track with mlflow
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, classification_test_metric)

            # Save the model and preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            model_wrapper = Model(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=model_wrapper)
            save_object("final_model/model.pkl", best_model)


            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        try:
            mlflow.set_tracking_uri("http://localhost:5001")
            with mlflow.start_run():
                mlflow.log_metric("f1_score", classificationmetric.f1_score)
                mlflow.log_metric("precision", classificationmetric.precision_score)
                mlflow.log_metric("recall", classificationmetric.recall_score)

                mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
               train_arr[:, :-1],                 # Features (all columns except last)
               train_arr[:, -1].astype(int),     # Target column 
               test_arr[:, :-1],
               test_arr[:, -1].astype(int),

               )


            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
