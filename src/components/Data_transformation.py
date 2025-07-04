import sys 
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Year', 'Mileage', 'Cylinders']
            categorical_columns = [
                "Make",
                "Model",
                "Body_Type",
                "Fuel_Type",
                "Location",
                "Transmission",
                "Color"
            ]
                
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
                
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])
                
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
                 
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
                               
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
                
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
             
            logging.info("Read the train and test data")
            print("Train DataFrame shape:", train_df.shape)
            print("Test DataFrame shape:", test_df.shape)
             
            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Price"
             
            if target_column_name not in train_df.columns:
                raise CustomException(f"'{target_column_name}' not found in train data", sys)
            if target_column_name not in test_df.columns:
                raise CustomException(f"'{target_column_name}' not found in test data", sys)

            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_features_train_df = train_df[[target_column_name]]
             
            input_features_test_df = test_df.drop(columns=[target_column_name])
            target_features_test_df = test_df[[target_column_name]]

            # DEBUGGING shapes
            print("Shape of input_features_train_df before transform:", input_features_train_df.shape)
            print("Shape of target_features_train_df before reshape:", target_features_train_df.shape)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()
            
            print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            print("input_feature_test_arr shape:", input_feature_test_arr.shape)

            # Ensure target is reshaped to column vector
            target_train_arr = np.array(target_features_train_df).reshape(-1, 1)
            target_test_arr = np.array(target_features_test_df).reshape(-1, 1)
            
            print("target_train_arr shape:", target_train_arr.shape)
            print("target_test_arr shape:", target_test_arr.shape)

            print("Shape after transform - X_train:", input_feature_train_arr.shape)
            print("Shape after reshape - y_train:", target_train_arr.shape)
            
            if input_feature_train_arr.ndim == 1:
                input_feature_train_arr = input_feature_train_arr.reshape(-1, 1)
            if target_train_arr.ndim == 1:
                target_train_arr = target_train_arr.reshape(-1, 1)


            train_arr = np.hstack((input_feature_train_arr, target_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_test_arr))

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
             
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )            
             
        except Exception as e:
            raise CustomException(e, sys)
