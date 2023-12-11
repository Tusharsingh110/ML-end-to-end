import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'

            model= load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            logging.info("Predict pipelining done successfully!!")
            return preds
        
        except Exception as e:
            logging.info("Prediction failed")
            raise CustomException(e,sys)

        

class CustomData:
    #maps input from the form to the backend
    def __init__(
        self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education,
        lunch:int,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int,):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender" :[self.gender],
                "race_ethnicity" :[self.race_ethnicity],
                "parental_level_of_education" :[self.parental_level_of_education],
                "lunch" :[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            logging.info("Converted to dataframe successfully")
            df=pd.DataFrame(custom_data_input_dict)
            print(df.head(1))
            return df

        except Exception as e:
            logging.info("Conversion to dataframe failed!!")
            raise CustomException(e,sys)



        

