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
# Utils File
# ==============================================================

import os
import sys
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# ==============================================================
# Save Object Function
# ==============================================================

def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



# ==============================================================
# Evaluate Models Function
# ==============================================================

def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    try:

        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]
            param = params[list(models.keys())[i]]


            # Hyperparameter tuning
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)


            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)


            # Scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)


            report[list(models.keys())[i]] = test_score


        return report


    except Exception as e:
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