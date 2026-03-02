import os 
# Used to work with file paths (like saving model into artifacts folder)

import sys
# Used for system-level error handling
# Helps in passing system information to CustomException

from dataclasses import dataclass
# @dataclass is used to create configuration classes easily
# It automatically creates constructor (__init__) for us


# Importing different ML models
# These are regression models because we are predicting math_score (continuous value)

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
# r2_score is used to evaluate regression model performance

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor


from src.exception import CustomException
# Custom exception class to handle errors professionally

from src.logger import logging
# Used to log execution steps in log file

from src.utils import save_object, evaluate_models
# save_object → saves trained model
# evaluate_models → evaluates multiple models and returns their scores



# ---------------------------------------------
# Configuration Class
# ---------------------------------------------

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    # Defines where trained model will be saved
    # artifacts/model.pkl



# ---------------------------------------------
# Model Trainer Class
# ---------------------------------------------

class ModelTrainer:
    
    def __init__(self):
        # Constructor runs automatically when object is created
        
        self.model_trainer_config = ModelTrainerConfig()
        # Creating configuration object
        # Now we can access trained_model_file_path



    def initiate_model_trainer(self, train_array, test_array):
        # This function trains multiple models and selects the best one
        
        try:
            logging.info("Split training and test input data")
            # Logging that splitting has started

            
            # Splitting features and target
            # train_array and test_array contain:
            # [features + target]
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],   # All columns except last → Features
                train_array[:,-1],    # Last column → Target
                test_array[:,:-1],    # Test features
                test_array[:,-1],     # Test target
            )


            # Dictionary of different regression models
            # Key → Model name
            # Value → Model object
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier" : CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostRegressor(),
            } 


            # Calling evaluate_models function
            # This trains all models and returns their test R2 scores
            model_report:dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
            )


            # Getting highest score from dictionary
            best_model_score = max(sorted(model_report.values()))
            # model_report.values() gives all R2 scores
            # max() selects highest score


            # Finding model name corresponding to best score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Getting actual model object
            best_model = models[best_model_name]


            # If best score is less than 0.6 → raise error
            if best_model_score < 0.6:
                raise CustomException("No best model found")


            logging.info(f"Best found model on both training and test dataset")
            # Logging successful model selection


            # Saving best model as .pkl file
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )


            # Predicting on test data using best model
            predicted = best_model.predict(X_test)


            # Calculating final R2 score
            r2_square = r2_score(y_test, predicted)


            # Returning final performance score
            return r2_square


        except Exception as e:
            # If any error occurs
            
            raise CustomException(e, sys)
            # Raise custom error for better debugging