import os
import sys
import dill
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score



from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    report = {}
    best_model_objects = {}

    for model_name, model in models.items():
        try:
            logging.info(f"Training and tuning model: {model_name}")
            params_grid = param.get(model_name, {})

            if params_grid:  # Only tune if parameters exist
                search = RandomizedSearchCV(
                    model,
                    params_grid,
                    n_iter=10,               # Try only 10 random combinations
                    cv=3,                    # 3-fold cross validation
                    verbose=0,
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            predictions = best_model.predict(X_test)
            score = r2_score(y_test, predictions)

            report[model_name] = score
            best_model_objects[model_name] = best_model

        except Exception as e:
            logging.warning(f"Error with model {model_name}: {e}")
            report[model_name] = 0
            best_model_objects[model_name] = model  # fallback to default

    return report, best_model_objects