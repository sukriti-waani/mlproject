# Data Ingestion is the process of collecting data from a source, reading it, and preparing it for further processing in the ML pipeline.

# Why Do We Create data_ingestion.py ?
# We create data_ingestion.py to collect, read, and prepare the dataset before training the model.
# It is the first step of the Machine Learning pipeline.



import os  # Used to work with file paths and folders (create folders, join paths, etc.)

import sys  # Used for system-level operations, mainly helpful in custom exception handling

import pandas as pd  # Pandas library is used to read and manipulate dataset (CSV files)

from dataclasses import dataclass  # Used to create a simple configuration class automatically

from sklearn.model_selection import train_test_split  # Used to split dataset into train and test sets

from src.exception import CustomException  # Your custom error handling class

from src.logger import logging  # Your logging file to print execution messages



@dataclass
class DataIngestionConfig:
    # This class stores all file paths related to data ingestion
    
    train_data_path: str = os.path.join('artifacts', "train.csv")
    # This creates path: artifacts/train.csv
    
    test_data_path: str = os.path.join('artifacts', "test.csv")
    # This creates path: artifacts/test.csv
    
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    # This creates path: artifacts/data.csv



class DataIngestion:
    # This class handles complete data ingestion process
    
    def __init__(self):
        # Constructor method runs automatically when object is created
        
        self.ingestion_config = DataIngestionConfig()
        # Creating object of configuration class
        # Now we can access train, test, raw file paths using self.ingestion_config



    def initiate_data_ingestion(self):
        # This is the main function that performs data ingestion
        
        logging.info("Entered the data ingestion component")
        # This logs a message saying ingestion has started
        
        try:
            # Try block means if any error happens, it will go to except block
            
            df = pd.read_csv(r'notebook\data\stud.csv')
            # Reads CSV file from given path
            # r before string avoids path error due to backslash
            
            logging.info("Dataset read successfully as pandas DataFrame")
            # Logging that file reading was successful


            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )
            # Creates the artifacts folder
            # os.path.dirname extracts folder name from path
            # exist_ok=True means no error if folder already exists


            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )
            # Saves original dataset as data.csv inside artifacts folder
            # index=False means row numbers will not be saved
            # header=True means column names will be saved
            
            logging.info("Raw data saved")
            # Logging raw data save completion


            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )
            # Splits dataset into two parts
            # 80% data → training
            # 20% data → testing
            # random_state=42 ensures same split every time
            
            logging.info("Train Test Split completed")
            # Logging split completion


            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )
            # Saves training dataset into train.csv
            

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )
            # Saves testing dataset into test.csv
            

            logging.info("Data Ingestion completed successfully")
            # Logging successful completion of ingestion


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            # Returns train and test file paths
            # These paths will be used in next pipeline component


        except Exception as e:
            # If any error happens in try block
            
            raise CustomException(e, sys)
            # Raise your custom exception
            # This helps in better error tracking in ML pipeline
            
            
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    


# run the below command in terminal to run this file
# python -m src.components.data_ingestion