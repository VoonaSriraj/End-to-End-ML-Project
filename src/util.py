import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves an object to the specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates models with GridSearchCV, avoiding the __sklearn_tags__ issue.
    """
    try:
        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            param_grid = param.get(model_name, {})

            try:
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                # Check if the model has been fitted
                check_is_fitted(best_model)

                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                model_report[model_name] = score
                logging.info(f"{model_name} R2 Score: {score:.4f}")

            except AttributeError as e:
                logging.error(f"Skipping {model_name} due to AttributeError: {str(e)}")
                continue

        return model_report

    except Exception as e:
        raise CustomException(f"Error in evaluate_models: {str(e)}", sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
