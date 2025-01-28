import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

import joblib

def save_object(file_path: str, obj: object) -> None:
    try:
        with open(file_path, 'wb') as file:
            joblib.dump(obj, file)
    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}", sys)
