# Data Ingestion is the process of collecting data from a source, reading it, and preparing it for further processing in the ML pipeline.

# Why Do We Create data_ingestion.py ?
# We create data_ingestion.py to collect, read, and prepare the dataset before training the model.
# It is the first step of the Machine Learning pipeline.



# ==============================================================
# Data Ingestion
# ==============================================================

# Data Ingestion is the process of collecting data from a source,
# reading it, and preparing it for further processing in the ML pipeline.

import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# ==============================================================
# Configuration Class
# ==============================================================

@dataclass
class DataIngestionConfig:
    
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# ==============================================================
# Data Ingestion Class
# ==============================================================

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion component")

        try:

            # Dataset path
            data_path = os.path.join("notebook", "data", "stud.csv")

            # Read dataset
            df = pd.read_csv(data_path)

            logging.info("Dataset read successfully")


            # Create artifacts folder if not exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )


            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Raw data saved")


            # Train Test Split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info("Train test split completed")


            # Save train data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )


            # Save test data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )


            logging.info("Data ingestion completed successfully")


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)



# ==============================================================
# Pipeline Execution
# ==============================================================

if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()


    # Data Transformation
    data_transformation = DataTransformation()

    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )


    # Model Training
    model_trainer = ModelTrainer()

    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

# run the below command in terminal to run this file
# python -m src.components.data_ingestion