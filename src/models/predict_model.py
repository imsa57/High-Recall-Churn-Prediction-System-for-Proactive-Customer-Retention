import numpy as np
import pandas as pd
import os, sys, json, mlflow , joblib
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List
from src.features.transformers import TenureBinningTransformer
from src.features.build_features import ChurnPreprocessor
from src.exception import CustomException
from src.logger import logger
import shap

_mlflow_model_uri = None
_mlflow_model=None


def get_model_prediction(model_pipeline:Pipeline, input:Dict[str, Any],threshold:float)->Dict[str,Any]:

    model_result = {}
    if model_pipeline is not None and input is not None:
        y_prob = model_pipeline.predict_proba(input)[0][1]

        if y_prob >= threshold:
            model_result['Churn']= "Yes"
        else:
            model_result['Churn'] = 'No'

        high_risk_features(model_pipeline, model_result, input=input)

        model_result['feature_value'] = input.to_dict(orient='records')
                
        return model_result

    else:
        CustomException("No model pipeline found", sys)

def high_risk_features(model_pipeline:Pipeline, model_result:Dict[str,str], input)->Dict[str,str]:   
         
        preprocessor = model_pipeline.steps[0][1]
        model = model_pipeline.steps[1][1]
        feature_names = preprocessor.get_feature_names_out()

        if model_pipeline.steps[1][0] == 'logisticregression':
            pass
        else:
            feature = preprocessor.transform(input)  
            explainer = shap.TreeExplainer(model)
            shap_value = explainer(feature)

            # shap_value.values shape is (1, 30, 2) -> we take index [0] for the row, all features [:,], and index [1] for class 1
            target_class_idx = 1 if model_result['Churn'] == 'Yes' else 0
            shap_vals_for_class = shap_value.values[0, :, target_class_idx]

            # 3. Create a clean DataFrame matching features to their SHAP values
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': shap_vals_for_class + shap_value.base_values[0][target_class_idx],
                'Abs_SHAP_Value': abs(shap_vals_for_class) # Used for magnitude ranking
            })
            top_risk_factors = None
            if target_class_idx == 1:
                top_risk_factors = feature_importance_df.sort_values(by='SHAP_Value', ascending=False).head(10)
            else:
                top_risk_factors = feature_importance_df.sort_values(by='SHAP_Value', ascending=True).head(10)


            parsed_features = []

            for feat in list(top_risk_factors['Feature']):
                if '_' in feat:
                    key, value = feat.split('_', 1)
                else:
                    key, value = feat, "Numerical Value"
                    
                parsed_features.append(key)
            model_result["Features"] = list(set(parsed_features))[0:6]
            return model_result


def get_best_model_deatil(file_path:str='models/all_models_metric.json')->dict:
    metrics = None
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        sorted_metric = sorted(metrics,key=lambda x : x['desiered_metrics']['precision'] , reverse=True)
        return sorted_metric[0]
    else:
        logger.error(f"No model found on path {file_path}")
        logger.info("Model training start............")


def predict_single(input:List[Any]) -> Dict[str,Any]:
    global _mlflow_model_uri
    global _mlflow_model

    feature_names = [
        'MonthlyCharges', 'tenure', 'Contract', 'PaymentMethod', 
        'SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'StreamingMovies', 'PaperlessBilling'
    ]
    model_detail = None
    try:
        model_detail = get_best_model_deatil()
        mlflow_model_url = model_detail['mlflow_model_path']
        if mlflow_model_url:
            try:
                if _mlflow_model_uri != mlflow_model_url:
                    _mlflow_model_uri = mlflow_model_url
                    _mlflow_model = mlflow.sklearn.load_model(mlflow_model_url)
                    logger.info(f'Model Loaded from mlflow: {mlflow_model_url}')


            except Exception as e:
                print("Local model used for production")
                _mlflow_model = joblib.load(model_detail['local_model_path'])
                # CustomException(e,sys)

        else:
            print("Local model used for production")
            _mlflow_model =joblib.load(model_detail['local_model_path'])
    except Exception as e:
        logger.error(f'Best Model info not found')


    threshold = model_detail['desiered_metrics']['threshold']

    X= input 
    if not isinstance(input, pd.DataFrame):
        X = pd.DataFrame([input], columns=feature_names)

    logger.info(f'Prediction started: {X}')
   
    result = get_model_prediction(model_pipeline=_mlflow_model, input=X,threshold=threshold)
    print(result)
    return result 