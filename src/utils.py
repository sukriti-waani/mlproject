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
# What is utils.py?
# ==============================================================

# utils.py is a helper file used in Machine Learning projects.
# It contains common reusable functions that can be used in many files.

# Instead of rewriting the same functions again and again
# in different files (like data_ingestion.py, model_trainer.py),
# we store them here and import them wherever needed.

# Example of functions we store here:
# 1. Saving trained models
# 2. Evaluating ML models
# 3. Loading objects
# 4. Common calculations

# This makes the project:
# ✔ Cleaner
# ✔ Easier to maintain
# ✔ More professional


# ==============================================================
# Importing Required Libraries
# ==============================================================

import os
# os module helps interact with the operating system.
# It is used here to:
# - create folders
# - join file paths
# - get directory names

import sys
# sys module provides system-specific parameters.
# It is mainly used here to pass system error information
# to our CustomException class.

import dill
# dill is used to save Python objects into files.
# It works like pickle but supports more complex objects
# such as functions, pipelines, etc.


from src.exception import CustomException
# Importing our custom exception class.
# This helps show detailed error messages
# when something goes wrong in the ML pipeline.


from sklearn.metrics import r2_score
# r2_score is a regression evaluation metric.
# It measures how well the model predictions match the actual values.

# R2 score range:
# 1 → Perfect prediction
# 0 → Average model
# <0 → Poor model


from sklearn.model_selection import GridSearchCV
# GridSearchCV is used for hyperparameter tuning.
# It tries multiple combinations of parameters
# and finds the best one for the model.


# ==============================================================
# Function: save_object
# ==============================================================

def save_object(file_path, obj):
    # file_path → location where we want to save the object
    # obj → object to be saved (model, pipeline, transformer, etc.)

    try:
        # Try block is used to catch errors safely.


        dir_path = os.path.dirname(file_path)
        # Extract directory path from the full file path.
        # Example:
        # file_path = artifacts/model.pkl
        # dir_path = artifacts


        os.makedirs(dir_path, exist_ok=True)
        # This creates the directory if it does not exist.
        # exist_ok=True means:
        # If the folder already exists, do not raise an error.


        with open(file_path, "wb") as file_obj:
            # Opening the file in write-binary mode.

            # w → write mode
            # b → binary mode (needed for saving objects)

            dill.dump(obj, file_obj)
            # dill.dump() converts the Python object into binary format
            # and stores it in the file.

            # Example:
            # model → converted to binary → stored in model.pkl


    except Exception as e:
        # If any error happens inside the try block,
        # this except block will run.

        raise CustomException(e, sys)
        # Raising a custom exception with system details.
        # This helps in debugging ML pipelines.


# ==============================================================
# Function: evaluate_models
# ==============================================================

def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    # This function trains multiple ML models
    # and evaluates their performance.

    # Inputs:

    # X_train → Training input features
    # y_train → Training output values
    # X_test → Testing input features
    # y_test → Testing output values

    # models → Dictionary of models
    # Example:
    # {
    #   "Random Forest": RandomForestRegressor(),
    #   "Linear Regression": LinearRegression()
    # }

    # params → Dictionary of hyperparameters for models
    # Example:
    # {
    #   "Random Forest": {"n_estimators":[50,100]}
    # }

    try:

        report = {}
        # Creating an empty dictionary.
        # This will store model names and their test scores.

        # Example result:
        # report = {
        #   "Random Forest": 0.87,
        #   "Linear Regression": 0.75
        # }


        for i in range(len(models)):
            # Loop through each model in the models dictionary.


            model = list(models.values())[i]
            # Extract the model object.

            # Example:
            # RandomForestRegressor()


            param = params[list(models.keys())[i]]
            # Extract the parameter grid for that model.


            # =====================================================
            # Hyperparameter Tuning
            # =====================================================

            gs = GridSearchCV(model, param, cv=3)
            # GridSearchCV tests multiple parameter combinations.

            # model → the machine learning model
            # param → dictionary of parameters
            # cv=3 → 3-fold cross validation


            gs.fit(X_train, y_train)
            # Train GridSearchCV to find the best parameters.


            model.set_params(**gs.best_params_)
            # Update the model with the best parameters found.


            model.fit(X_train, y_train)
            # Train the final model using best parameters.


            # =====================================================
            # Model Predictions
            # =====================================================

            y_train_pred = model.predict(X_train)
            # Model predicts output for training data.


            y_test_pred = model.predict(X_test)
            # Model predicts output for unseen test data.


            # =====================================================
            # Model Evaluation
            # =====================================================

            train_score = r2_score(y_train, y_train_pred)
            # Calculate R2 score for training data.


            test_score = r2_score(y_test, y_test_pred)
            # Calculate R2 score for testing data.


            report[list(models.keys())[i]] = test_score
            # Store the test score in dictionary.

            # Example:
            # report["Random Forest"] = 0.87


        return report
        # Return dictionary containing all model scores.


    except Exception as e:
        # If any error occurs during model evaluation.

        raise CustomException(e, sys)
        # Raise custom exception for debugging.
        
# Function to load a saved Python object (like a trained model or preprocessor)
# from a file stored on disk.

def load_object(file_path):
    # file_path → the location of the saved object
    # Example: artifacts/model.pkl or artifacts/preprocessor.pkl

    try:
        # Try block is used so that if any error occurs while loading
        # the object, we can catch it and handle it properly.

        # Opening the file in "rb" mode
        # r  → read mode
        # b  → binary mode (required for reading serialized objects)
        with open(file_path, "rb") as file_obj:

            # dill.load() reads the binary data from the file
            # and converts it back into the original Python object.
            #
            # Example:
            # model.pkl → converted back to trained ML model
            # preprocessor.pkl → converted back to preprocessing pipeline
            return dill.load(file_obj)

    except Exception as e:
        # If any error occurs (file not found, corrupted file, etc.)
        # the exception will be caught here.
        raise CustomException(e, sys)
    
    
    
    
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