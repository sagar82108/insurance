import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class predictpipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path= os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('error occured in data ingestion config')        
            raise CustomException(e,sys)


class customdata:
    def __init__(self,
                 age:int,
                 bmi:float,
                 sex:object,
                 smoker:object,
                 region:object):
        
        self.age=age
        self.bmi=bmi
        self.sex=sex
        self.smoker=smoker
        self.region=region
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'age': [self.age],  # Making single values as lists
                'bmi': [self.bmi],
                'sex': [self.sex],
                'smoker': [self.smoker],
                'region': [self.region]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('dataframe gathered')
            return df
        
        except Exception as e:
            logging.info('exception occured in pipeline')        
            raise CustomException(e,sys)
    