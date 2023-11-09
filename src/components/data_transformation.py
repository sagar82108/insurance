from sklearn.impute import SimpleImputer ##handling missing values
from sklearn.preprocessing import StandardScaler ## handling feature scaling
from sklearn.preprocessing import OneHotEncoder ## encoding categorical features

##pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os,sys
from dataclasses import  dataclass

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')





class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('data transformation initiated')
            categorical_cols=['sex','smoker','region']
            numerical_cols=['age','bmi']

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse=False)),
                ('scaler', StandardScaler())
            ])

            # Preprocessor ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', categorical_pipeline, categorical_cols)
            ])

            return preprocessor  # returning the configured preprocessor

        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)

    


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            logging.info('read train and test data completed')
            logging.info(f'train dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'test dataframe head : \n{test_df.head().to_string()}')
            logging.info('obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()
            

            X_train=train_df.drop(columns=['expenses'],axis=1)
            y_train=train_df['expenses']

            X_test=test_df.drop(columns=['expenses'],axis=1)
            y_test=test_df['expenses']
            logging.info(f'train dataframe head: \n{X_train.head().to_string()}')
            logging.info(f'test dataframe head : \n{y_train.head().to_string()}')
            

            #applying the transformation

            
            x_train_array = pd.DataFrame(
            preprocessing_obj.fit_transform(X_train),
            columns=preprocessing_obj.get_feature_names_out()
             )
            x_test_array = pd.DataFrame(
            preprocessing_obj.transform(X_test),
            columns=preprocessing_obj.get_feature_names_out()
            )



            logging.info("applying preprocesssing object on training and testing datasets")

            train_arr=np.c_[x_train_array,np.array(y_train)]
            test_arr=np.c_[x_test_array,np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessor pickle is created and logged")
            logging.info(f'train dataframe head: \n{train_arr}')
            logging.info(f'test dataframe head : \n{test_arr}')

            return (
                train_arr,
                test_arr,
                
            )
                
        except Exception as e:
            raise CustomException(e,sys)
