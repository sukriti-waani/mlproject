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
            
            # Dictionary that stores hyperparameters for all regression models
            params = {

                # -------------------- Decision Tree --------------------
                "Decision Tree": {

                    # Different ways to measure error while splitting nodes
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],

                    # Strategy used to choose split at each node
                    "splitter": ["best", "random"],

                    # Number of features to consider while splitting
                    # sqrt = √(total features)
                    # log2 = log2(total features)
                    "max_features": ["sqrt", "log2"]
                },

                # -------------------- Random Forest --------------------
                "Random Forest": {

                    # Error measurement methods
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],

                    # Number of features considered at each split
                    # None = use all features
                    "max_features": ["sqrt", "log2", None],

                    # Number of trees in the forest
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                # -------------------- Gradient Boosting --------------------
                "Gradient Boosting": {

                    # Loss function used to calculate error
                    "loss": ["squared_error", "huber", "absolute_error", "quantile"],

                    # Step size while updating model
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],

                    # Fraction of data used for training each tree
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],

                    # Method to measure quality of split
                    "criterion": ["squared_error", "friedman_mse"],

                    # Number of features to consider while splitting
                    "max_features": ["sqrt", "log2"],

                    # Number of boosting stages (trees)
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                # -------------------- Linear Regression --------------------
                "Linear Regression": {
                    # No hyperparameters defined (default model settings)
                },

                # -------------------- K-Neighbors Regressor --------------------
                "K-Neighbors Regressor": {

                    # Number of nearest neighbors used for prediction
                    "n_neighbors": [5, 7, 9, 11]

                    # weights -> uniform (equal weight) or distance (closer = more weight)
                    # algorithm -> method to search neighbors (optional tuning)
                },

                # -------------------- XGBoost Regressor --------------------
                "XGBRegressor": {

                    # Step size shrinkage (controls overfitting)
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],

                    # Number of boosting trees
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                # -------------------- CatBoost Regressor --------------------
                "CatBoost Regressor": {

                    # Depth of each tree
                    "depth": [6, 8, 10],

                    # Learning rate
                    "learning_rate": [0.01, 0.05, 0.1],

                    # Number of boosting iterations
                    "iterations": [30, 50, 100]
                },

                # -------------------- AdaBoost Regressor --------------------
                "AdaBoost Regressor": {

                    # Learning rate for updating weights
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],

                    # Loss function used in boosting
                    "loss": ["linear", "square", "exponential"],

                    # Number of weak learners (usually small trees)
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }
            
            
            # Call the evaluate_models function
            model_report = evaluate_models(

                X_train=X_train,   # Training input features (used to train models)

                y_train=y_train,   # Training target/output values

                X_test=X_test,     # Testing input features (used to check performance)

                y_test=y_test,     # Testing target/output values

                models=models,     # Dictionary containing all model objects

                params=params      # Dictionary containing hyperparameter grids
            )

            # model_report will return a dictionary like:
            # {
            #   "Random Forest": 0.89,
            #   "XGBRegressor": 0.92,
            #   ...
            # }
            # (model name → model score)


            # Find the model name that has the highest score
            best_model_name = max(model_report, key=model_report.get)

            # max() checks values using model_report.get
            # It returns the key (model name) with highest value


            # Get the score of that best model
            best_model_score = model_report[best_model_name]

            # Access dictionary using best_model_name as key


            # Print best model name
            print("Best Model:", best_model_name)

            # Print best model score
            print("Best Score:", best_model_score)


            # Calling evaluate_models function
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