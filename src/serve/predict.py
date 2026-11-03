import os, sys, json
from typing import Dict, Any
from src.logger import logger
from src.models.predict_model import predict_single, get_best_model_deatil
from main import main as pipeline_main

def get_model_info(file_path:str='models/all_models_metric.json')->bool:

    if os.path.exists(file_path) and os.path.getsize(file_path) != 0:
        model_info = get_best_model_deatil(file_path)
        logger.info(f'Model found for predicton.')
        return True
    else:
        logger.error(f"No model found on path {file_path}")
        logger.info("Model training start............")
        pipeline_main()



def predict_churn(input:Dict[str,Any])->Dict[str,str]:
        input_data = json.loads(input)
        input = []
        for key, value in  input_data.items(): 
            input.append(value)
        prediction_data = predict_single(input=input)
        return prediction_data

