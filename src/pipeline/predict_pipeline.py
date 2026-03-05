# Importing sys module
# sys provides system-specific parameters and functions
# We mainly use it here to pass error details to CustomException
import sys

# Importing os module
# os helps in working with file paths and directories
# Example: joining folder names like artifacts/model.pkl
import os

# Importing pandas library
# Pandas is used for working with data in table format (DataFrame)
import pandas as pd

# Importing CustomException class
# This is our custom error handling class
# It helps display meaningful error messages in the ML pipeline
from src.exception import CustomException

# Importing load_object function from utils.py
# This function is used to load saved objects (model.pkl, preprocessor.pkl)
from src.utils import load_object



# ===============================================================
# Prediction Pipeline Class
# ===============================================================

class PredictPipeline:
    
    # Constructor method
    # It runs automatically when the class object is created
    def __init__(self):
        pass
        # No initialization needed here



    # Prediction function
    # This function takes input features and returns predicted result
    def predict(self, features):
        
        try:
            
            # Creating path to the trained model file
            # artifacts/model.pkl contains the trained ML model
            model_path = os.path.join("artifacts", "model.pkl")
            
            
            # Creating path to the preprocessing object
            # artifacts/preprocessor.pkl contains the transformation pipeline
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            
            # Just printing for debugging
            print("Before Loading")
            
            
            # Loading trained ML model
            model = load_object(file_path=model_path)
            
            
            # Loading preprocessing pipeline
            preprocessor = load_object(file_path=preprocessor_path)
            
            
            # Debug message to confirm objects loaded
            print("After Loading")
            
            
            # Applying preprocessing transformation on input features
            # Example transformations:
            # - Encoding categorical features
            # - Scaling numeric values
            data_scaled = preprocessor.transform(features)
            
            
            # Passing transformed data into trained model
            # Model predicts output (math score)
            preds = model.predict(data_scaled)
            
            
            # Returning prediction result
            return preds
        
        except Exception as e:
            # If any error occurs, raise custom exception
            raise CustomException(e, sys)




# ===============================================================
# Custom Data Class
# ===============================================================

# This class is used to convert user input (from web form)
# into a pandas DataFrame so the model can use it

class CustomData:
    
    # Constructor function
    # It receives input values from user form
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):

        # Assigning values to object variables

        self.gender = gender
        # Example value: Male / Female

        self.race_ethnicity = race_ethnicity
        # Example: group A / group B / group C

        self.parental_level_of_education = parental_level_of_education
        # Example: Bachelor's degree

        self.lunch = lunch
        # Example: standard / free/reduced

        self.test_preparation_course = test_preparation_course
        # Example: none / completed

        self.reading_score = reading_score
        # Numeric score between 0-100

        self.writing_score = writing_score
        # Numeric score between 0-100



    # ===============================================================
    # Convert Input Data into DataFrame
    # ===============================================================

    def get_data_as_data_frame(self):
        
        try:
            
            # Creating dictionary from user input
            # Each value is placed inside a list
            # because pandas DataFrame expects list-like values
            
            custom_data_input_dict = {
                
                "gender": [self.gender],
                
                "race_ethnicity": [self.race_ethnicity],
                
                "parental_level_of_education": [self.parental_level_of_education],
                
                "lunch": [self.lunch],
                
                "test_preparation_course": [self.test_preparation_course],
                
                "reading_score": [self.reading_score],
                
                "writing_score": [self.writing_score],
            }


            # Converting dictionary into pandas DataFrame
            # This format is required by the ML pipeline
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            
            # Handling errors using custom exception
            raise CustomException(e, sys)