import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

#since we need to predict the marks which are continuous values we need regression algorithms
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Model Trainer Initiated...") 
            X_train, y_train, X_test, y_test = (
                #take all cols except the last one
                train_arr[:,:-1],
                #take last col as train target
                train_arr[:,-1],
                #take all cols except the last one
                test_arr[:,:-1],
                #take last col as test target
                test_arr[:,-1],
            ) 

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,models = models)
            #To get best model score
            best_model_score = max(sorted(model_report.values()))

            # To get the best model from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found!!")

            logging.info(f"Best found model on both training and testing dataset is:  {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            print(model_report)
            return r2_square


        except Exception as e:
            logging.info("Model Training/Testing Failed!!")
            raise CustomException(e,sys)

    


