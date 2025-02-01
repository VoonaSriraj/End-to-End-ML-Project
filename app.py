import sys
import pandas as pd
import os
from flask import Flask, request, render_template
from src.exception import CustomException
from src.util import load_object

# Flask App Setup
application = Flask(__name__)
app = application

# Define PredictPipeline class
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

# Define CustomData class
class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str,
                 test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

# Home Route
@app.route('/')
def index():
    return render_template('home.html')

# Prediction Route
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Collect data from the form
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score = float(request.form.get('reading_score'))
        writing_score = float(request.form.get('writing_score'))

        # Create an instance of CustomData
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # Convert data to DataFrame for prediction
        features = data.get_data_as_data_frame()

        # Create a PredictPipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(features)

        # Return the result to the template
        return render_template('home.html', results=str(result[0]))

    except Exception as e:
        # Handle exceptions and return an error message
        return f"Error during prediction: {e}", 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
