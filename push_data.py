import os  
import sys  
import json  

# Load environment variables from a .env file
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables into the runtime

# Retrieve MongoDB connection URL from environment variables
MONGO_DB_URL = os.getenv("MONGO_DB_URL")  
print(MONGO_DB_URL)  # Print the MongoDB URL to confirm it is loaded correctly

# Import the certifi package to verify SSL certificates when connecting to MongoDB
import certifi
ca = certifi.where()  # Returns the path to the bundled CA certificates for secure connections


import pandas as pd  
import numpy as np  
import pymongo  


from src.exception.exception import CustomException  
from src.logging.logger import logging  

class DataExtract():
    def __init__(self):
        """
        Constructor for NetworkDataExtract class.
        """
        try:
            pass  
        except Exception as e:
            # Raise a custom exception with details about the error
            raise CustomException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            # Read the CSV file into a Pandas DataFrame
            data = pd.read_csv(file_path)
            
            # Reset the DataFrame index to avoid index issues
            data.reset_index(drop=True, inplace=True)
            
            # Convert DataFrame to JSON, transpose it, and extract values as a list of records
            records = list(json.loads(data.T.to_json()).values())
            
            # Return the list of JSON-like records
            return records
        except Exception as e:
            
            raise CustomException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            # Store parameters for database, collection, and records
            self.database = database
            self.collection = collection
            self.records = records

            # Establish a connection to MongoDB using the connection URL
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL,serverSelectionTimeoutMS=5000)
            
            # Access the specified database
            self.database = self.mongo_client[database]
            
            # Access the specified collection within the database
            self.collection = self.database[collection]
            
            # Insert all records into the MongoDB collection
            result=self.collection.insert_many(records)
            
            # Return the number of records inserted
            return len(result.inserted_ids)
        except Exception as e:
            
            raise CustomException(e, sys)

if __name__ == '__main__':
    """
    Main block to execute the program.
    Reads a CSV file, converts it to JSON, and inserts the data into MongoDB.
    """
    # Define file path, database, and collection names
    FILE_PATH = "dataset/brustkrebs_wisconsin_data.csv"  # Path to the CSV file
    DATABASE = "Dataset"  # Name of the MongoDB database
    Collection = "Brustkrebs"  # Name of the MongoDB collection

    # Create an instance of the NetworkDataExtract class
    obj = DataExtract()

    # Convert CSV data to a list of JSON-like records
    records = obj.csv_to_json_convertor(file_path=FILE_PATH)
    

    # Insert the records into the specified MongoDB database and collection
    no_of_records = obj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)  # Print the number of records inserted



