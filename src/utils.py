# ==============================================================
# What is utils.py?
# ==============================================================

# utils.py stands for "utility functions file".
# Utility functions are helper functions that are used
# in multiple places in the project.

# Why do we create utils.py?

# In a Machine Learning project, many small reusable tasks are needed:
# - Saving objects (models, preprocessors)
# - Loading objects
# - Evaluating models
# - Calculating metrics
# - Common helper operations

# Instead of writing these functions again and again
# in every file (data_ingestion.py, data_transformation.py, model_trainer.py),
# we keep them in a separate file called utils.py.

# Benefits:
# ✔ Code becomes clean
# ✔ Reusable functions
# ✔ Professional project structure
# ✔ Easier maintenance
# ✔ Industry standard practice

# In this file, we are creating a function
# to save Python objects (like models or preprocessors)
# into a file using dill library.


# ==============================================================
# Importing Required Libraries
# ==============================================================

import os
# Used for working with file paths and directories
# Example: creating folders

import sys
# Used for system-level error handling
# Helps in custom exception tracking

import dill
# dill is used to serialize (save) Python objects
# It is similar to pickle but more powerful
# It can save models, functions, pipelines, etc.

import numpy as np 
# Imported but not used here (can be removed if not needed)

import pandas as pd
# Imported but not used here (can be removed if not needed)

from src.exception import CustomException
# Custom exception class used to handle errors properly

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# ==============================================================
# Function: save_object
# ==============================================================

def save_object(file_path, obj):
    # file_path → where we want to save the object
    # obj → the object we want to save (model, preprocessor, etc.)
    
    try:
        # Extracting directory path from file path
        # Example:
        # If file_path = artifacts/preprocessor.pkl
        # dir_path = artifacts
        dir_path = os.path.dirname(file_path)
        
        
        # Creating the directory if it does not exist
        # exist_ok=True means:
        # If folder already exists, do not show error
        os.makedirs(dir_path, exist_ok=True)
        
        
        # Opening the file in "write binary" mode
        # "wb" means:
        # w → write
        # b → binary (needed for saving objects)
        with open(file_path, "wb") as file_obj:
            
            # Using dill to dump (save) the object into file
            # This converts Python object into byte stream
            # and stores it inside .pkl file
            dill.dump(obj, file_obj)
            
            # What Is Serialization?
                # When we do:
                # dill.dump(obj, file_obj)
                # We are converting a Python object into a storable format.

                # Example:
                # Model → Convert into file → Save as .pkl
                # Later we can load it and use for prediction.
            
            
    except Exception as e:
        # If any error happens while saving
        # Raise custom exception with system info
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    # This function evaluates multiple ML models.
    # It takes:
    # X_train → Training input features
    # y_train → Training target values
    # X_test → Testing input features
    # y_test → Testing target values
    # models → Dictionary of model_name : model_object
    
    try:
        # Try block is used to catch any errors during model training or evaluation
        
        report = {}
        # Creating an empty dictionary
        # This will store model name and its test score
        # Example:
        # report = {
        #     "Random Forest": 0.85,
        #     "Linear Regression": 0.78
        # }
        

        for i in range(len(list(models))):
            # Looping through dictionary of models
            # model_name → name of model (string)
            # model → actual ML model object
            # Example:
            # model_name = "Random Forest"
            # model = RandomForestRegressor()
            
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            # Train model
            # model.fit(X_train, y_train)
            # .fit() trains the model using training data
            # The model learns patterns from X_train and y_train

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            # Model predicts output for training data
            # Used to check how well model learned training data

            y_test_pred = model.predict(X_test)
            # Model predicts output for test data
            # This shows how well model performs on unseen data


            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            # Calculating R2 score for training data
            # R2 score measures how good predictions are
            # Value range:
            # 1 → Perfect prediction
            # 0 → Average model
            # Negative → Poor model

            test_model_score = r2_score(y_test, y_test_pred)
            # Calculating R2 score for test data
            # Test score is more important because it checks real performance


            report[list(models.keys())[i]] = test_model_score
            # Storing test score in dictionary
            # Key = model name
            # Value = test R2 score


        return report
        # After loop finishes,
        # Return dictionary containing all models and their scores


    except Exception as e:
        # If any error occurs during training or evaluation
        
        raise CustomException(e, sys)
        # Raise your custom exception
        # Helps track detailed error in ML pipeline
    
    
    
# ============================================================
# What is Serialization?
# ============================================================

# Serialization means converting a Python object into a file
# so that we can store it permanently and reuse it later.

# Example of objects we serialize in ML:
# - Trained Machine Learning model
# - Preprocessing pipeline
# - Custom function
# - ColumnTransformer object

# Why do we need serialization?

# During training:
# Model is created in memory (RAM).
# But once program stops, memory clears.
# So we save model into a file (like .pkl file).

# Later during deployment:
# We load that saved file and use it for prediction.
# No need to retrain the model again.


# ============================================================
# What is pickle?
# ============================================================

# pickle is a built-in Python module.
# It is used to serialize (save) and deserialize (load) Python objects.

# Example usage:

# import pickle
# pickle.dump(model, file)

# pickle works well for:
# - Simple objects
# - Basic ML models
# - Lists, dictionaries, arrays

# Limitation:
# It cannot properly handle complex objects
# like lambda functions or some advanced pipelines.


# ============================================================
# What is dill?
# ============================================================

# dill is an external library (needs installation using pip install dill).
# It extends pickle and supports more complex objects.

# Example usage:

# import dill
# dill.dump(model, file)

# dill can serialize:
# - Functions
# - Lambda functions
# - Complete pipelines
# - Complex Python objects
# - Closures


# ============================================================
# Why We Use Dill in ML Projects?
# ============================================================

# In ML projects, we often save:
# - Preprocessing pipeline
# - ColumnTransformer
# - Custom feature engineering functions
# - Trained model

# These objects may internally contain complex structures.

# So dill is safer because:
# ✔ It reduces serialization errors
# ✔ It handles more object types
# ✔ It is more flexible


# ============================================================
# Explanation
# ============================================================

# Pickle is a built-in Python module used for serializing
# and deserializing Python objects.
# Dill is an extended version of pickle that can serialize
# more complex Python objects such as functions and lambda expressions.
# Dill is more flexible but slightly slower than pickle.