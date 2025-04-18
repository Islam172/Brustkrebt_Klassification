import yaml 
from src.exception.exception import CustomException
from src.entity.artifact_entity import ClassificationMetricArtifact
from src.logging.logger import logging
import os, sys
import pickle
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score,precision_score,recall_score


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)     
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file. Optionally replaces the file if it already exists.
    
    Parameters:
        file_path (str): Path to the YAML file.
        content (object): Data to be written to the file.
        replace (bool): Whether to replace the file if it exists. Default is False.
    """
    try:
        if replace and os.path.exists(file_path):  # Check if the file should be replaced
            os.remove(file_path)  # Remove the existing file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create parent directories if needed
        with open(file_path, "w") as file:
            yaml.dump(content, file)  # Write the content to the YAML file
    except Exception as e:
        raise CustomException(e, sys)    
    


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.
    
    Parameters:
        file_path (str): Path to save the file.
        obj (object): The object to serialize and save.
    """
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create parent directories if needed
        with open(file_path, "wb") as file_obj:  # Open the file in binary write mode
            pickle.dump(obj, file_obj)  # Serialize and save the object
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using pickle.
    
    Parameters:
        file_path (str): Path to the file containing the object.
    
    Returns:
        object: The deserialized Python object.
    """
    try:
        if not os.path.exists(file_path):  # Check if the file exists
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:  # Open the file in binary read mode
            return pickle.load(file_obj)  # Deserialize and return the object
    except Exception as e:
        raise CustomException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a NumPy array from a file.
    
    Parameters:
        file_path (str): Path to the file containing the NumPy array.
    
    Returns:
        np.array: The loaded NumPy array.
    """
    try:
        with open(file_path, "rb") as file_obj:  # Open the file in binary read mode
            return np.load(file_obj, allow_pickle=True)  # Load and return the NumPy array
    except Exception as e:
        raise CustomException(e, sys)    
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning 
    and calculates R² scores for training and testing datasets.
    
    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Testing feature set.
        y_test (pd.Series): Testing target set.
        models (dict): Dictionary of model names and instances.
        param (dict): Dictionary of hyperparameters for each model.
    
    Returns:
        dict: Dictionary containing test R² scores for each model.
    """
    try:
        report = {}

        for i in range(len(list(models))):  # Iterate over models
            model = list(models.values())[i]  # Get the model instance
            para = param[list(models.keys())[i]]  # Get hyperparameters for the model

            # Perform GridSearchCV for hyperparameter tuning
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Update model with the best hyperparameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train the model

            # Predict for train and test datasets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores for training and testing datasets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save the test score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report  # Return the dictionary of test scores
    except Exception as e:
        raise CustomException(e, sys)



def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculates classification metrics (F1 score, recall, and precision) for a given set of true and predicted values.

    Parameters:
        y_true (array-like): True labels (ground truth).
        y_pred (array-like): Predicted labels from the model.

    Returns:
        ClassificationMetricArtifact: An object containing the F1 score, precision, and recall.
    """
    try:
        average_type = "binary" if len(np.unique(y_true)) == 2 else "macro"

        model_f1_score = f1_score(y_true, y_pred, average=average_type)
        model_precision_score = precision_score(y_true, y_pred, average=average_type)
        model_recall_score = recall_score(y_true, y_pred, average=average_type)


        # Create a classification metric artifact object to encapsulate the metrics
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score, 
            recall_score=model_recall_score
        )

        # Return the classification metric artifact
        return classification_metric
    except Exception as e:
        # Raise a custom exception in case of errors
        raise CustomException(e, sys)
