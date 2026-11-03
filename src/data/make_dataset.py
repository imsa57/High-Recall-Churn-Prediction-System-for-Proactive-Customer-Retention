# -*- coding: utf-8 -*-
import click, sys, os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.logger import logger
from src.exception import CustomException
import pandas as pd, numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

'''
Creating Loader class
load_data
validate data
    checking Required data
    Checking columns datatype
    check range of columns
    drop duplicate
    check messing value
    

'''

class DataLoader:
    def __init__(self, data_path:str= "data/raw/Telco-Customer-Churn.csv"):
        self.data_path = Path(data_path)
        self.df = None
        self.X_train= None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_data(self)-> pd.DataFrame:

        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded sucessfully. shape {self.df.shape}")
            return self.df
        
        except Exception as e:
            logger.error(f"Error in Loading data : {str(e)}")
            raise CustomException(e,sys)

        
    def validate_data(self,df:pd.DataFrame)->pd.DataFrame:
        # checking required columns
        required_columns = ["Churn",'MonthlyCharges','SeniorCitizen', 'Partner', 'Dependents', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection','StreamingMovies', 'PaperlessBilling', 'PaymentMethod','tenure','Contract']
        missing_columns =[]
        
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        try:
            if len(missing_columns) != 0:
                raise ValueError(f'Missing columns Found {len(missing_columns)} {missing_columns}',sys)   
        except Exception as e:
                raise CustomException(f'Missing columns Found {len(missing_columns)} {missing_columns}',sys)   
        

        logger.info("No missing columns found")


        # Deleting Extra Columns
        extra_columns = []
        for col in df.columns:
            if col not in required_columns:
                extra_columns.append(col)
        if len(extra_columns) > 0:
            logger.info(f"Some extra columns found.. {extra_columns}")
            df.drop(columns=extra_columns,inplace=True)

        
        # checking data type  
        columns_dtype = {
            "MonthlyCharges": "float64",
            "SeniorCitizen": "str",
            "Partner": "str",
            "Dependents": "str",
            "InternetService": "str",
            "OnlineSecurity": "str",
            "OnlineBackup": "str",
            "DeviceProtection": "str",
            "StreamingMovies": "str",
            "PaperlessBilling": "str",
            "PaymentMethod": "str",
            "tenure": "int64",
            "Contract": "str",
            "Churn":"str"
        }
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

        logger.info('Data type checking start...')
        for col in df.columns:
            try:
                if df[col].dtype != columns_dtype[col]:
                    raise
            except Exception as e:
                logger.error(f'Data type {col} is not matching.')
                raise CustomException(e,sys)
        logger.info("Data type checking Done!")

        # dropping dulicates
        logger.info(f'Checking Duplicate value : Found {df.duplicated().sum()} duplicate value.')
        df.drop_duplicates(inplace=True)

        logger.info(f'Cheking null value : Found {df.isnull().sum().sum()} null value.')
        self.df = df
        return self.df

    def split_data(self, test_size:float=0.2, random_state:int=42)->Tuple[pd.DataFrame]:

        if self.df is None:
            raise ValueError('Df not found. Please check')

        logger.info(f'Splitting data into test_size={test_size}, random_state={random_state}')

        X = self.df.drop(columns=['Churn'])
        y = self.df['Churn'].map({'No':0 ,'Yes':1})

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=random_state)

        logger.info(f'Splitting done --> train_set:{self.X_train.shape}, test_set: {self.X_test.shape}')
        return  self.X_train, self.X_test, self.y_train, self.y_test



# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data.')
    loader = DataLoader()
    df = loader.load_data()
    validated_df = loader.validate_data(df)
    return loader.split_data(test_size=0.2,random_state=42)


# if __name__ == '__main__':
#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]
    
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
