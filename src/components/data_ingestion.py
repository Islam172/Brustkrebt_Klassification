from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

import os  
import sys  
import numpy as np 
import pandas as pd  
import pymongo 
from typing import List  
from sklearn.model_selection import train_test_split  
from dotenv import load_dotenv  # For loading environment variables from a .env file

load_dotenv()  # Loads environment variables from the .env file
MONGO_DB_URL = os.getenv("MONGO_DB_URL")  # Retrieves the MongoDB connection URL from environment variables

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the configuration for data ingestion.

        Parameters:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)  # Raises a custom exception on error
    
    def export_collection_as_dataframe(self):
        """
        Reads data from a MongoDB collection and returns it as a Pandas DataFrame.
        """
        try:
            # Get database and collection names from the configuration
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            # Connect to the MongoDB client
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            
            # Access the specified collection
            collection = self.mongo_client[database_name][collection_name]
            
            # Convert the MongoDB collection into a Pandas DataFrame
            df = pd.DataFrame(list(collection.find()))
            
            # Drop the "_id" column if it exists
            if "_id" in df.columns.to_list():
                df = df.drop(columns=(["_id","Unnamed: 32","id"]), axis=1)
            
            # Replace "na" strings with NaN
            df.replace({"na": np.nan}, inplace=True)
            
            return df  # Return the DataFrame
        except Exception as e:
            raise CustomException(e, sys)  # Handle exceptions with a custom exception
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Exports the DataFrame into a feature store by saving it as a CSV file.

        Parameters:
            dataframe (pd.DataFrame): DataFrame to be exported.

        Returns:
            pd.DataFrame: The input DataFrame.
        """
        try:
            # Get the path to the feature store file from the configuration
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
            # Create the directory for the feature store if it doesn't exist
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save the DataFrame to the feature store as a CSV file
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            return dataframe  # Return the DataFrame
        except Exception as e:
            raise CustomException(e, sys)  # Handle exceptions with a custom exception
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the DataFrame into training and testing datasets and saves them to files.

        Parameters:
            dataframe (pd.DataFrame): DataFrame to be split into train and test sets.
        """
        try:
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train-test split on the DataFrame.")

            # Create directories for saving train and test data
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the train and test datasets to CSV files
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Exported train and test datasets to files.")
        except Exception as e:
            raise CustomException(e, sys)  # Handle exceptions with a custom exception

    def initiate_data_ingestion(self):
        """
        Orchestrates the data ingestion process:
        1. Reads data from MongoDB.
        2. Saves data to a feature store.
        3. Splits data into train and test sets.

        Returns:
            DataIngestionArtifact: Artifact containing paths to the train and test datasets.
        """
        try:
            # Step 1: Export data from MongoDB to a DataFrame
            dataframe = self.export_collection_as_dataframe()
            
            # Step 2: Save the DataFrame to the feature store
            dataframe = self.export_data_into_feature_store(dataframe)
            
            # Step 3: Split the data into train and test sets
            self.split_data_as_train_test(dataframe)
            
            # Create an artifact with paths to the train and test datasets
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return dataingestionartifact  # Return the artifact
        except Exception as e:
            raise CustomException(e, sys)  # Handle exceptions with a custom exception

            


        except Exception as e:
            raise CustomException(e,sys)