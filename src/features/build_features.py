import sys, os
from src.logger import logger
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
from src.exception import CustomException
from src.data.make_dataset import main as data_main
from .transformers import TenureBinningTransformer

def create_preprocessing_pipeline(df:pd.DataFrame, model_type:str ='linear')->ColumnTransformer:
    continuous_feature = ['MonthlyCharges']
    categorical_feature = [
    'SeniorCitizen', 'Partner', 'Dependents', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'StreamingMovies', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    if model_type == 'linear':
            
        num_pipeline=make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler()
        )
        num_pipeline=make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler()
        )
        tenure_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            TenureBinningTransformer(),
            OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
            MinMaxScaler()
        )

        contract_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            MinMaxScaler()
        )

        categorical_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        )


    else:
            
        num_pipeline=make_pipeline(
            SimpleImputer(strategy='mean'),
        )
        num_pipeline=make_pipeline(
            SimpleImputer(strategy='mean'),
        )
        tenure_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            TenureBinningTransformer(),
            OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
        )

        contract_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        )

        categorical_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        )


    preprosessing =ColumnTransformer(
        transformers=[
            ("nf",num_pipeline,continuous_feature),
            ('tf',tenure_pipeline,['tenure']),
            ('cf',contract_pipeline,['Contract']),
            ('cgf',categorical_pipeline,categorical_feature)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    logger.info('Preprocessing pipeline created successfully')
    return preprosessing


def create_full_pipeline(df:pd.DataFrame,model,model_type:str ='linear')->Pipeline:

    preprocessor = create_preprocessing_pipeline(df,model_type)

    full_pipeline = make_pipeline(
        preprocessor,
        model
    )
    return full_pipeline


def get_feature_names_from_pipeline(pipeline:ColumnTransformer)->list:
    try:
        feature_names = pipeline.get_feature_names_out()
        return list(feature_names)
    except Exception as e:
        logger.warning(f'Could not get features names: {str(e)}')
        return []


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,model_type='linear'):
        self.model_type = model_type
        self.preprocessor = None
        self.features_name_in = None

    def fit(self, X , y=None):
        self.features_name_in = X.columns.tolist()

        self.preprocessor = create_preprocessing_pipeline(
            X,
            model_type=self.model_type
        )

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise CustomException("Preprocessor not fitted. Call fit() first",sys)        

        return self.preprocessor.transform(X)

    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X,y)
        return self.transform(X)

    def get_feature_names_out(self):
        if self.preprocessor is None:
            raise CustomException("Preprocessor not fitted. Call fit() first.",sys)

        return get_feature_names_from_pipeline(self.preprocessor)



if __name__ == "__main__":

    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "heart.csv"
    print("Started")
    X_train,X_test,y_train,y_test = data_main()
    print(X_train.shape)
    if X_train is not None:
        
        preprocessor = ChurnPreprocessor()
        X_transformed = preprocessor.fit_transform(X_train)
        print(X_transformed.shape)
        print(f"Feature names: {preprocessor.get_feature_names_out()[:10]}...")

    else:
        print("Sample data not found!")    
