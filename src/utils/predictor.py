from src.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys


from src.exception.exception import CustomException
from src.logging.logger import logging


class Model:
    """
    A class that encapsulates a machine learning model and its preprocessor. 
    Provides functionality for making predictions using the preprocessed data.
    """
    def __init__(self, preprocessor, model):
        """
        Initializes the NetworkModel with a preprocessor and a trained model.

        Parameters:
            preprocessor (object): The preprocessing pipeline for input data.
            model (object): The trained machine learning model.
        """
        try:
            self.preprocessor = preprocessor  # Save the preprocessor object.
            self.model = model  # Save the trained model object.
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, x):
        """
        Makes predictions using the model and the preprocessor.
        """
        try:
            # Transform the raw input data using the preprocessor.
            x_transform = self.preprocessor.transform(x)
            
            # Predict the output using the transformed data and the model.
            y_hat = self.model.predict(x_transform)
            
            # Return the predicted values.
            return y_hat
        except Exception as e:
            raise CustomException(e, sys)
