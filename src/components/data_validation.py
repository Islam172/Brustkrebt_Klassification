from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from src.utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    """
    DataValidation is responsible for:
    1. Validating the structure of the datasets (e.g., column count).
    2. Detecting dataset drift using statistical tests.
    3. Generating drift reports.
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation class with required configurations.
        
        Parameters:
            data_ingestion_artifact (DataIngestionArtifact): Contains paths for train and test data.
            data_validation_config (DataValidationConfig): Configuration for data validation.
        """
        try:
            # Initialize class variables
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
            # Load schema configuration from the schema.yaml file
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            # Raise custom exception on error
            raise CustomException(e, sys)

    @staticmethod   #Makes the method independent of class instances.
    def read_data(file_path) -> pd.DataFrame:
     """
    Reads data from a CSV file and returns it as a Pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
         pd.DataFrame: The loaded dataset in a Pandas DataFrame format.
    """ 
     try:
        # Read the CSV file into a Pandas DataFrame
        return pd.read_csv(file_path)
     except Exception as e:
        # Raise a custom exception if any error occurs while reading the file
        raise CustomException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
     """
    Validates whether the DataFrame has the required number of columns.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to validate.

    Returns:
        bool: True if the column count matches the expected number, False otherwise.
    """
     try:
        # Access the 'columns' key in the schema configuration
        required_columns = self._schema_config.get("columns", [])
        # Get the expected number of columns from the schema configuration
        number_of_columns = len(required_columns)

        # Log the required number of columns and the actual column count in the DataFrame
        logging.info(f"Required number of columns: {number_of_columns}")
        logging.info(f"Data frame has columns: {len(dataframe.columns)}")
        
        # Check if the actual number of columns matches the expected number
        if len(dataframe.columns) == number_of_columns:
            return True  # Column count is valid
        return False  # Column count does not match
     except Exception as e:
        # Raise a custom exception if any error occurs during validation
        raise CustomException(e, sys)
    


    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
     """
    Detects dataset drift between two datasets using the Kolmogorov-Smirnov test.
    
    Parameters:
        base_df (pd.DataFrame): The reference (base) dataset.
        current_df (pd.DataFrame): The current dataset to compare with the base dataset.
        threshold (float): The p-value threshold for drift detection. Default is 0.05.
    
    Returns:
        bool: True if no significant drift is detected, False otherwise.
     """
     try:
        # Initial drift detection status (True means no drift detected)
        status = True

        # Dictionary to store drift details for each column
        report = {}

        # Iterate through each column in the base dataset
        for column in base_df.columns:
            # Extract column data from base and current datasets
            d1 = base_df[column]
            d2 = current_df[column]

            # Perform Kolmogorov-Smirnov test to compare distributions
            is_same_dist = ks_2samp(d1, d2)

            # Check if the p-value is below the threshold
            if threshold <= is_same_dist.pvalue:
                is_found = False  # No drift detected
            else:
                is_found = True  # Drift detected
                status = False  # Update overall status to indicate drift

            # Update the report with drift details for this column
            report.update({
                column: {
                    "p_value": float(is_same_dist.pvalue),  # Store the p-value
                    "drift_status": is_found               # Store drift status
                }
            })

        # Path to save the drift report
        drift_report_file_path = self.data_validation_config.drift_report_file_path

        # Create the directory for the drift report if it doesn't exist
        dir_path = os.path.dirname(drift_report_file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Write the drift report to a YAML file
        write_yaml_file(file_path=drift_report_file_path, content=report)

        # Return the overall drift detection status
        return status
     except Exception as e:
        # Raise a custom exception with error details
        raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
     """
    Orchestrates the entire data validation process:
    1. Reads train and test datasets.
    2. Validates the number of columns in each dataset.
    3. Detects dataset drift between the train and test datasets.
    4. Saves valid train and test datasets to disk.
    5. Creates and returns a DataValidationArtifact.
    
    Returns:
        DataValidationArtifact: Contains paths and status information of the validation process.
     """
     try:
        # Retrieve paths for train and test datasets
        train_file_path = self.data_ingestion_artifact.trained_file_path
        test_file_path = self.data_ingestion_artifact.test_file_path

        # Step 1: Read train and test datasets
        train_dataframe = DataValidation.read_data(train_file_path)
        test_dataframe = DataValidation.read_data(test_file_path)

        # Step 2: Validate the number of columns in the train dataset
        status = self.validate_number_of_columns(dataframe=train_dataframe)
        if not status:
            error_message = "Train dataframe does not contain all columns.\n"
            raise ValueError(error_message)

        # Step 2: Validate the number of columns in the test dataset
        status = self.validate_number_of_columns(dataframe=test_dataframe)
        if not status:
            error_message = "Test dataframe does not contain all columns.\n"
            raise ValueError(error_message)

        # Step 3: Detect dataset drift between train and test datasets
        status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

        # Step 4: Save valid train and test datasets to disk
        dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the valid train dataset
        train_dataframe.to_csv(
            self.data_validation_config.valid_train_file_path, index=False, header=True
        )

        # Save the valid test dataset
        test_dataframe.to_csv(
            self.data_validation_config.valid_test_file_path, index=False, header=True
        )

        # Step 5: Create a DataValidationArtifact to store validation results
        data_validation_artifact = DataValidationArtifact(
            validation_status=status,  # Overall validation status
            valid_train_file_path=self.data_ingestion_artifact.trained_file_path,  # Path to valid train data
            valid_test_file_path=self.data_ingestion_artifact.test_file_path,  # Path to valid test data
            invalid_train_file_path=None,  # No invalid train data in this process
            invalid_test_file_path=None,  # No invalid test data in this process
            drift_report_file_path=self.data_validation_config.drift_report_file_path,  # Path to drift report
        )

        # Return the validation artifact
        return data_validation_artifact
     except Exception as e:
        # Raise a custom exception with error details
        raise CustomException(e, sys)
  
