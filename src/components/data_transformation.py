from sklearn.impute import SimpleImputer ##handling missing values
from sklearn.preprocessing import StandardScaler ## handling feature scaling
from sklearn.preprocessing import OneHotEncoder ## encoding categorical features

##pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os,sys
from dataclasses import  dataclass

import numpy as numpy
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformation:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')





class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation()

    def get_data_transformation_object(self, numerical_cols, categorical_cols):
        try:
            logging.info('data transformation initiated')

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

            X_train=train_df.drop(columns=['expenses'])
            y_train=train_df[['expenses']]

            X_test=test_df.drop(columns=['expenses'])
            y_train=test_df[['expenses']]

            #applying the transformation

            x_train_array=preprocessing_obj.fit_transform(X_train)
            x_test_array=preprocessing_obj.tranform(X_test)