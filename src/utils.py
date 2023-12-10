import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True) 

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        logging.info("Saving object into pickle failed!!")
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test, models):
    try:
        report = {}

        for _,model_name in enumerate(models):
            model=models[model_name]
            # print(model)
            #Train model
            model.fit(X_train,y_train)

            #predict on train data
            y_pred_train = model.predict(X_train)

            #predict on test data
            y_pred_test = model.predict(X_test)

            #get the r2 score from train data
            model_score_train = r2_score(y_train,y_pred_train)

            #get the r2 score from train data
            model_score_test = r2_score(y_test,y_pred_test)

            report[model_name]=model_score_test

        logging.info("Model Evaluation completed successfully!!")
        return report
        
    except Exception as e:
        logging.info("Model evaluation Failed!!")
        raise CustomException(e,sys)