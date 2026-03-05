# Importing Flask components
# Flask → framework used to build web applications
# request → used to get data sent from the HTML form
# render_template → used to render HTML pages
from flask import Flask, request, render_template


# Importing numpy library
# Used for numerical operations (not heavily used here but commonly imported in ML apps)
import numpy as np


# Importing pandas
# Used for working with tabular data (DataFrames)
import pandas as pd


# Importing StandardScaler from sklearn
# Used to scale numeric data (though not used directly here)
from sklearn.preprocessing import StandardScaler


# Importing our custom ML pipeline classes
# CustomData → converts user input into DataFrame
# PredictPipeline → loads model and predicts result
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



# Creating Flask application object
# __name__ tells Flask where to look for templates and static files
application = Flask(__name__)


# Creating another reference to the Flask app
# Some deployment platforms expect variable name "app"
app = application



# ============================================================
# Route for Home Page
# ============================================================

# @app.route('/') means this function runs when user visits:
# http://localhost:5000/
@app.route('/')

def index():
    # render_template loads an HTML file from templates folder
    # Here it loads templates/index.html
    return render_template('index.html')



# ============================================================
# Route for Prediction
# ============================================================

# This route handles prediction requests
# URL will be:
# http://localhost:5000/predictdata

# methods=['GET','POST']
# GET → user opens the page
# POST → user submits the form

@app.route('/predictdata', methods=['GET', 'POST'])

def predict_datapoint():

    # If user just opens the page
    if request.method == 'GET':

        # Show home.html page
        return render_template('home.html')



    # If user submits the form
    else:

        # Creating CustomData object
        # This collects all form inputs sent from HTML page

        data = CustomData(

            # Getting gender value from form
            gender = request.form.get('gender'),

            # Getting ethnicity value
            race_ethnicity = request.form.get('ethnicity'),

            # Getting parental education level
            parental_level_of_education = request.form.get('parental_level_of_education'),

            # Getting lunch type
            lunch = request.form.get('lunch'),

            # Getting test preparation course
            test_preparation_course = request.form.get('test_preparation_course'),

            # Getting reading score
            # Converting to float because model expects numeric input
            reading_score = float(request.form.get('reading_score')),

            # Getting writing score
            writing_score = float(request.form.get('writing_score'))

        )


        # Convert user input into pandas DataFrame
        # Model expects data in DataFrame format
        pred_df = data.get_data_as_data_frame()


        # Print input data in terminal (for debugging)
        print(pred_df)


        # Debug message before prediction
        print("Before Prediction")



        # Creating object of prediction pipeline
        predict_pipeline = PredictPipeline()


        # Debug message
        print("Mid Prediction")


        # Calling prediction function
        # This will:
        # 1. Load preprocessor.pkl
        # 2. Transform input features
        # 3. Load model.pkl
        # 4. Predict math score
        results = predict_pipeline.predict(pred_df)


        # Debug message after prediction
        print("after Prediction")


        # Sending prediction result back to webpage
        # results[0] → because prediction returns array like [78.4]
        return render_template('home.html', results=results[0])



# ============================================================
# Main Function
# ============================================================

# This runs the Flask app only if this file is executed directly
if __name__ == "__main__":

    # Starting Flask server

    # host="0.0.0.0" means the app is accessible from any network
    # Default port = 5000

    app.run(host="0.0.0.0")