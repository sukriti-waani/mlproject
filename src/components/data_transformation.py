# 🌟 What is Data Transformation?
# Data Transformation is the process of converting raw data into a clean and structured format so that a machine learning model can understand it.

#  Why Do We Create data_transformation.py File?
# In ML pipeline, we divide work into steps:
# Data Ingestion → Load data
# Data Transformation → Clean & convert data
# Model Training → Train model
# Model Evaluation → Check performance

# This file is responsible for:
# ✔ Handling missing values
# ✔ Converting categorical data into numbers
# ✔ Scaling numerical data
# ✔ Saving preprocessing object for future use

#  Why Is Transformation Important?
# Machine Learning models:
# ❌ Cannot understand text (like "male", "female")
# ❌ Cannot handle missing values
# ❌ Perform poorly if data is not scaled
# So we transform data before giving it to the model.



# Importing sys module
# Used for handling exceptions and system-related error information
import sys  

# Importing os module
# Used to work with file paths and directories
import os 

from dataclasses import dataclass  
#  What is @dataclass?
    # @dataclass is a Python decorator used to create classes that mainly store data.
    # It automatically:
    #     Creates constructor
    #     Stores variables
    #     Makes code clean and short
    
    #     Without dataclass
    # class DataTransformationConfig:
    #     def __init__(self):
    #         self.preprocessor_obj_file_path = "artifacts/preprocessor.pkl"
    
    #     With dataclass (Cleaner)
    # @dataclass
    # class DataTransformationConfig:
    #     preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

    # ✔ Less code
    # ✔ More readable
    # ✔ Professional approach



# Importing pandas
# Used to read and manipulate CSV datasets
import pandas as pd  

# Importing numpy
# Used for numerical operations and array handling
import numpy as np  

# ColumnTransformer allows applying different transformations
# to different columns (numerical & categorical separately)
from sklearn.compose import ColumnTransformer  

# Pipeline allows combining multiple steps into a single workflow
# Example: Imputer → Scaler
from sklearn.pipeline import Pipeline   

# SimpleImputer is used to fill missing values
from sklearn.impute import SimpleImputer  

# OneHotEncoder converts categorical text into numbers
# StandardScaler scales numerical values
from sklearn.preprocessing import OneHotEncoder, StandardScaler  

# CustomException is your own error-handling class
from src.exception import CustomException  

# Logging is used to record execution messages
from src.logger import logging  

# save_object is used to save preprocessing object as a .pkl file
from src.utils import save_object  


# --------------------------------------------
# Configuration class using dataclass
# --------------------------------------------

@dataclass
class DataTransformationConfig:
    # This defines where we will save the preprocessing object
    # os.path.join ensures correct file path formation
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


# --------------------------------------------
# Main Data Transformation Class
# --------------------------------------------

class DataTransformation:
    
    # Constructor method
    # Runs automatically when object is created
    def __init__(self):
        # Creating object of configuration class
        # So we can access file path inside this class
        self.data_transformation_config = DataTransformationConfig()

    
    # This function creates preprocessing object
    def get_data_transformer_object(self):
        try:
            # Defining numerical columns
            # These contain numbers and need scaling
            numerical_columns = ["writing_score", "reading_score"]

            # Defining categorical columns
            # These contain text values and need encoding
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Creating numerical pipeline
            # Step 1: Fill missing values using median
            # Step 2: Scale the data
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Creating categorical pipeline
            # Step 1: Fill missing values using most frequent category
            # Step 2: Convert text to numbers using OneHotEncoder
            # Step 3: Scale encoded values
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Logging column information
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # ColumnTransformer applies:
            # num_pipeline → numerical columns
            # cat_pipeline → categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            # Return the complete preprocessing object
            return preprocessor

        except Exception as e:
            # If any error occurs, raise custom exception
            raise CustomException(e, sys)

    
    # This function performs full transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading training dataset
            train_df = pd.read_csv(train_path)

            # Reading testing dataset
            test_df = pd.read_csv(test_path)

            # Logging successful read
            logging.info("Train and test data loaded successfully")

            # Getting preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Defining target column (output variable)
            target_column_name = "math_score"

            # Dropping target column from train input features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)

            # Extracting target column from training data
            target_feature_train_df = train_df[target_column_name]

            # Dropping target column from test input features
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            # Extracting target column from testing data
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing data")

            # Fit preprocessing object on training data
            # Fit means learning missing values, encoding structure, scaling parameters
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Transform test data using same fitted object
            # Important: Never fit on test data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed input features and target into single array
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            # Same combination for test data
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            # Saving preprocessing object as pickle file
            # This will be used later during prediction
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Returning processed train data, test data, and path of preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # If any error occurs during transformation
            raise CustomException(e, sys)