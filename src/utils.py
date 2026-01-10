import os
import sys
import pickle
import json
import joblib
from tensorflow import keras

from src.exception import CustomException

def save_pkl(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_pkl(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_keras(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file_obj:
            model.save(file_obj)

    except Exception as e:
        raise CustomException(e, sys)    

def load_keras(file_path):
    try:
        keras.model.load_model(file_path)
    except Exception as e:
        raise CustomException(sys, e)

def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file_obj:
            json.dump(obj, file_obj, indent=4)

    except Exception as e:
        raise CustomException(e, sys)

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)