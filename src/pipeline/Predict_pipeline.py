import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Make : str,
        Model : str,
        Body_Type: str,
        Fuel_Type: str,
        Location: str,
        Year: int,
        Mileage: int,
        Cylinders : int,
        Color: str,
        Transmission: str,
        
        ):
        
        self.Model = Model
        
        self.Make = Make

        self.Body_Type = Body_Type

        self.Fuel_Type = Fuel_Type

        self.Location = Location

        self.Year= Year
        
        self.Cylinders= Cylinders
        
        self.Color= Color

        self.Mileage = Mileage

        self.Transmission = Transmission

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Model": [self.Model],
                "Make": [self.Make],
                "Mileage": [self.Mileage],
                "Year": [self.Year],
                "Body_Type": [self.Body_Type],
                "Fuel_Type": [self.Fuel_Type],
                "Transmission": [self.Transmission],
                "Location": [self.Location],
                "Color": [self.Color],
                "Cylinders": [self.Cylinders],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)