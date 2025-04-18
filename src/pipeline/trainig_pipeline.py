import os
import sys

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from src.constants.training_pipeline import TRAINING_BUCKET_NAME
from src.cloud.s3_sync import S3Sync
from src.constants.training_pipeline import SAVED_MODEL_DIR


class TrainingPipeline:
    """
    Class to orchestrate the machine learning pipeline, including data ingestion,
    validation, transformation, model training, and syncing artifacts to S3.
    """
    def __init__(self):
        """
        Initializes the training pipeline with configurations and S3 syncer.
        """
        self.training_pipeline_config = TrainingPipelineConfig()  # Load pipeline configurations.
        self.s3_sync = S3Sync()  # Initialize the S3 sync utility.


    def start_data_ingestion(self):
        """
        Initiates the data ingestion process to fetch and prepare raw data.

        Returns:
            DataIngestionArtifact: Contains paths to ingested train and test data.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data ingestion process")
            
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        Initiates the data validation process to check data quality and schema.

        Parameters:
            data_ingestion_artifact (DataIngestionArtifact): Paths to ingested train and test data.

        Returns:
            DataValidationArtifact: Contains paths to validated data and validation reports.
        """
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            
            logging.info("Initiate data validation process")
            data_validation_artifact = data_validation.initiate_data_validation()
            
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
     
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        """
        Transforms the validated data into a format suitable for machine learning models.

        Parameters:
            data_validation_artifact (DataValidationArtifact): Paths to validated train and test data.

        Returns:
            DataTransformationArtifact: Contains paths to transformed train and test data.
        """
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Trains machine learning models using transformed data.

        Parameters:
            data_transformation_artifact (DataTransformationArtifact): Paths to transformed train and test data.

        Returns:
            ModelTrainerArtifact: Contains paths to the trained model and performance metrics.
        """
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def sync_artifact_dir_to_s3(self):
        """
        Syncs local artifacts (data and models) to an S3 bucket for storage.
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise CustomException(e, sys)



    def sync_saved_model_dir_to_s3(self):
        """
        Syncs the local final trained model to an S3 bucket for deployment and sharing.
        """
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise CustomException(e, sys)
     

    def run_pipeline(self):
        """
        Executes the entire pipeline: ingestion, validation, transformation, training, and syncing artifacts.

        Returns:
            ModelTrainerArtifact: The result of the trained model, including its metrics.
        """
        try:
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            
            # Step 3: Data Transformation
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            
            # Step 4: Model Training
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
            # Step 5: Sync artifacts and models to S3
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
      
      
