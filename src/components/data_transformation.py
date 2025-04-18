import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import CustomException 
from src.logging.logger import logging
from src.utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    This class handles the data transformation process, including:
    1. Imputation of missing values.
    2. Standardization of input features.
    3. Saving the transformed data and preprocessing pipeline.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Returns a preprocessing pipeline with:
        - KNNImputer for missing values
        - StandardScaler for standardization
        """
        try:
            imputer : KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            scaler = StandardScaler()
            processor : Pipeline = Pipeline([
                ("imputer", imputer),
                ("scaler", scaler)
            ])
            return processor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Orchestrates the data transformation process, including:
        1. Loading valid train and test data.
        2. Imputing missing values and transforming the datasets.
        3. Saving transformed data and the preprocessing object.

        Returns:
            DataTransformationArtifact: Contains paths to transformed data and the preprocessing object.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            # Step 1: Load validated datasets
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Step 2: Separate input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].map({'M': 1, 'B': 0})

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].map({'M': 1, 'B': 0})

            # Step 3: Preprocessing pipeline
            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(input_feature_train_df)

            transformed_train_input = preprocessor.transform(input_feature_train_df)
            transformed_test_input = preprocessor.transform(input_feature_test_df)

            # Step 4: Combine with target
            train_arr = np.c_[transformed_train_input, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test_input, np.array(target_feature_test_df)]

            # Step 5: Save data and pipeline object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            # Optional: save to model directory as well
            save_object("final_model/preprocessor.pkl", preprocessor)

            # Step 6: Return artifact
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
