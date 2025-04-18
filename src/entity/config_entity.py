from datetime import datetime

# Import the os module to work with file paths and directories
import os

# Import constants related to the training pipeline from a custom module
from src.constants import training_pipeline

# Print pipeline name and artifact directory for debugging or verification
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

# Define a configuration class for the training pipeline
class TrainingPipelineConfig:
    """
    This class is used to configure the training pipeline,
    including paths and directories for artifacts and models.
    """
    def __init__(self, timestamp=datetime.now()):
        # Format the timestamp to a specific format for directory naming
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        
        # Name of the pipeline
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        
        # Root directory for all artifacts
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        
        # Path to the artifact directory for this pipeline run
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        
        # Directory to store the final model
        self.model_dir = os.path.join("final_model")
        
        # Store the formatted timestamp as a string
        self.timestamp: str = timestamp




# Define a configuration class for the data ingestion stage
class DataIngestionConfig:
    """
    This class handles the configuration for the data ingestion process,
    including paths to raw data, feature store, and split datasets.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for data ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )
        
        # Path to the feature store file
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )
        
        # Path to the training dataset file
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )
        
        # Path to the testing dataset file
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )
        
        # Ratio for splitting the dataset into training and testing
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        
        # Name of the collection in the database for data ingestion
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        
        # Name of the database for data ingestion
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME

        
        # Ratio for splitting the dataset into training and testing
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        

class DataValidationConfig:
    """
    This class handles the configuration for validating data,
    including directories for valid and invalid datasets and drift reports.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for data validation artifacts
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        
        # Directory for valid datasets
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        
        # Directory for invalid datasets
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        
        # Paths for valid training and testing datasets
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        
        # Paths for invalid training and testing datasets
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        
        # Path for the drift report
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


class DataTransformationConfig:
    """
    This class handles the configuration for transforming data,
    including paths for transformed datasets and preprocessing objects.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for data transformation artifacts
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        
        # Path for the transformed training dataset (stored as .npy files)
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        
        # Path for the transformed testing dataset (stored as .npy files)
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )
        
        # Path for the preprocessing object used for transformation
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    """
    This class handles the configuration for training the model,
    including paths for saving the trained model and expected metrics.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for model training artifacts
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        
        # Path to save the trained model
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, training_pipeline.MODEL_FILE_NAME
        )
        
        # Minimum expected accuracy for the model
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        
        # Threshold for determining overfitting or underfitting
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
