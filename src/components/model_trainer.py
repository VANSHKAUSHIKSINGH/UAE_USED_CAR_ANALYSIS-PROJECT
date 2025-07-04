import os
import sys
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import  (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid, GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging   
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test data')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1], 
                test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'Linear Regression': LinearRegression()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor": {
                    'depth': [6, 8, 10, 12],
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'iterations': [30,50,100]
                },
                "AdaBoostRegressor": {
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }       
            }

            logging.info("Starting model evaluation and tuning")

            model_report = {}
            best_model_objects = {}

            for model_name, model in models.items():
                try:
                    logging.info(f"Training and tuning model: {model_name}")
                    params_grid = params.get(model_name, {})
                    param_combinations = list(ParameterGrid(params_grid))
                    if len(param_combinations) <= 10:
                        search = GridSearchCV(model, params_grid, cv=3, n_jobs=-1, verbose=0)
                    else:
                        n_iter = min(10, len(param_combinations))
                        search = RandomizedSearchCV(model, params_grid, n_iter=n_iter, cv=3, n_jobs=-1, verbose=0)

                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_

                    predictions = best_model.predict(X_test)
                    score = r2_score(y_test, predictions)

                    model_report[model_name] = score
                    best_model_objects[model_name] = best_model

                except Exception as e:
                    logging.warning(f"Model {model_name} failed: {e}")
                    model_report[model_name] = 0

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = best_model_objects[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)
