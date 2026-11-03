import sys,os, joblib, json
from pathlib import Path 
from datetime import datetime
import  pandas as pd
import numpy as np
from src.logger import logger
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, List
from scipy.stats import uniform, randint
from src.features.build_features import create_full_pipeline
import mlflow
import dagshub




class ModelTrainer:
    def __init__(self,random_state:int=42,cv_fold:int=5,model_dir:str='models'):
        self.random_state = random_state
        self.cv_fold =cv_fold
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models = {}
        self.training_results = {}
        self.best_model = None
        self.best_model_name=None
        self.best_model_metrics = None
        self.time_stamp = None


    def get_all_available_model(self)->Dict[str,Any]:
        models={
            'LogisticRegression': LogisticRegression(),
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier()
        }
        params_grid={
            'LogisticRegression':{
                'logisticregression__penalty': ['l1', 'l2'],
                'logisticregression__C': uniform(0.01, 10.0),       # Ranges from 0.01 to 10.01
                'logisticregression__class_weight': ['balanced', None],
                'logisticregression__l1_ratio': uniform(0.0, 1.0),  # Required when penalty='elasticnet'
                'logisticregression__solver': ['liblinear']
            },
            'RandomForest':{
                'randomforestclassifier__n_estimators': [100, 150, 200, 250, 300],
                'randomforestclassifier__max_depth': [10, 15, 20, 30, None],
                'randomforestclassifier__min_samples_split': randint(2, 20), # Integer uniform distribution
                'randomforestclassifier__min_samples_leaf': randint(1, 15),
                'randomforestclassifier__class_weight': ['balanced', 'balanced_subsample']
            },
            'XGBoost':{
                'xgbclassifier__n_estimators': [100, 200, 300],
                'xgbclassifier__max_depth': [3, 5, 7],
                'xgbclassifier__learning_rate': uniform(0.01, 0.2),  # Ranges from 0.01 to 0.21
                'xgbclassifier__subsample': uniform(0.6, 0.4),       # Ranges from 0.6 to 1.0
                'xgbclassifier__scale_pos_weight': uniform(1.0, 3.0)
            }
        }
        return models, params_grid


    def train_model(
            self,
            model_name:str,
            model,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            X_test:pd.DataFrame,
            y_test:pd.Series,
            reference_df:pd.DataFrame,
            param_grid:dict
    )->Dict[str, Any]:
        logger.info(f'Training {model_name}..........')

        model_type = 'linear'

        if model_name == 'XGBoost' or model_name == 'RandomForest':
            model_type = 'non-linear'

        pipeline = create_full_pipeline(
            reference_df,
            model,
            model_type=model_type
        )
        try:

            with mlflow.start_run(run_name=f'{self.time_stamp}_{model_name}_Tuning') as run:
                random_search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_grid, 
                    cv=self.cv_fold,
                    n_iter=1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                random_search.fit(X_train,y_train)

                # Make prediction

                y_pred_train = random_search.best_estimator_.predict(X_train)
                y_pred_test = random_search.best_estimator_.predict(X_test)

                # Get probability scores for AUC metrics
                y_score_train = random_search.best_estimator_.predict_proba(X_train)[:, 1]
                y_score_test = random_search.best_estimator_.predict_proba(X_test)[:, 1]

                
                # Calculate training metrics
                train_metrics = {
                    'accuracy': round(accuracy_score(y_train, y_pred_train),4),
                    'precision': round(precision_score(y_train, y_pred_train, zero_division=0),4),
                    'recall': round(recall_score(y_train, y_pred_train, zero_division=0),4),
                    'f1': round(f1_score(y_train, y_pred_train, zero_division=0),4),
                    'roc_auc': round(roc_auc_score(y_train, y_score_train),4),
                    'pr_auc': round(average_precision_score(y_train, y_score_train),4),
                }
                
                # Calculate test metrics
                test_metrics = {
                    'accuracy': round(accuracy_score(y_test, y_pred_test),4),
                    'precision': round(precision_score(y_test, y_pred_test, zero_division=0),4),
                    'recall': round(recall_score(y_test, y_pred_test, zero_division=0),4),
                    'f1': round(f1_score(y_test, y_pred_test, zero_division=0),4),
                    'roc_auc': round(roc_auc_score(y_test, y_score_test),4),
                    'pr_auc': round(average_precision_score(y_test, y_score_test),4),
                }     


                # Compile results
                best_estimator = random_search.best_estimator_
                model_name = best_estimator.steps[1][0]
                results = {
                    'model_name': model_name,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'training_timestamp': datetime.now().isoformat()
                }

                self.trained_models[model_name] = best_estimator
                self.training_results[model_name] = results

                logger.info(f"{model_name} training completed. Test PR_AUC: {test_metrics['pr_auc']:.4f}")
                
                flattened_metrics = {}
                for key, value in train_metrics.items():
                    flattened_metrics[f'train_{key}'] = value
                for key, value in test_metrics.items():
                    flattened_metrics[f'test_{key}'] = value

                mlflow.log_metrics(flattened_metrics)
                mlflow.log_params({
                    **random_search.best_params_
                })
            mlflow.end_run()    
                # return results
        except Exception as e:
            raise CustomException(e,sys)



    def train_all_model(
            self,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            X_test:pd.DataFrame,
            y_test:pd.Series,
            reference_df:pd.DataFrame
    )->Dict[str,Dict[str, Any]]:

        # dagshub and mlflow
        # dagshub.init(repo_owner='imsa57', repo_name='telcho-churn-prediction', mlflow=True)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:7006")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('Churn prediction 4')

        self.time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

        logger.info("Starting training of all models...")
        models , params_grid = self.get_all_available_model()

        for model_name, model in models.items():
            try:
                self.train_model(
                    model_name=model_name,
                    model=model,
                    param_grid=params_grid[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    reference_df=reference_df
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue   

        # Select best model based on test ROC-AUC
        self._select_best_model(X_train=X_train,y_train=y_train)
        self.save_reference_profile(reference_df=reference_df)
        
        logger.info(f"Training completed. Best model: {self.best_model_name}")


    def _select_best_model(self,X_train:pd.DataFrame,y_train:pd.Series, metric:str='roc_auc'):

        results = []
        for name,pipeline in self.trained_models.items():
            y_score = pipeline.predict_proba(X_train)[:,1]
            precisions, recalls, thresholds = precision_recall_curve(y_train, y_score)
            id_90_recall = np.argmax(np.sort(recalls) >= 0.90)
            recall = round(recalls[id_90_recall],3)
            precision = round(precisions[id_90_recall],3)
            threshold = round(thresholds[id_90_recall],3)
            results.append({"name":name,"recall":recall, 'precision':precision,'threshold':threshold})

        self._select_best_recall_model(results)
        


    def _select_best_recall_model(self,results:List[dict]):
            if not results:
                logger.warning("No training results available for best model selection")
                return
            
            with mlflow.start_run(run_name=f'{self.time_stamp}_Best_model_selection') as run:

                df = pd.DataFrame(results)
                filtered_df = df[df['recall'] > 0.90]
                selected_model = filtered_df.sort_values(by=['precision'], ascending=False)
                best_modal_metric = selected_model.iloc[0,:]

                best_recall_model = self.trained_models[selected_model.iloc[0,:]['name']].steps[1]

                self.best_model_name = best_recall_model[0]

                self.best_model = self.trained_models[self.best_model_name]

                best_training_metrics = self.training_results[self.best_model_name]

                best_training_metrics['desiered_metrics'] = best_modal_metric.to_dict()
                self.best_model_metrics = best_training_metrics

                artifact_path_mlflow = f"{self.best_model_name }_model_{self.time_stamp}"
                mlflow.sklearn.log_model(
                    sk_model=self.best_model,
                    name=artifact_path_mlflow,
                    serialization_format="skops",
                    skops_trusted_types=[
                        "src.features.transformers.TenureBinningTransformer",
                        "numpy.dtype",
                        "sklearn.tree._tree.Tree"
                    ]
                )
                best_model_path = f"runs:/{run.info.run_id}/{artifact_path_mlflow}"
                self.best_model_metrics['mlflow_model_path'] = best_model_path

                mlflow.log_params({
                    **best_training_metrics['desiered_metrics']
                })
            mlflow.end_run()    
            logger.info(f'Best model selected: {best_modal_metric}')
            self.save_model()

    def _select_best_precision_model(self):
        pass


    def save_model(self)->str:

        model_name= self.best_model_name
        best_pipeline = self.best_model

        timestamp = self.time_stamp
        filename = f'{model_name}_{timestamp}.pkl'

        model_dir = os.path.join(self.model_dir,f'{model_name}_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        registry_file_path =  os.path.join(f'{self.model_dir}', 'all_models_metric.json')

        model_path = self.model_dir / f'{model_name}_{timestamp}' / filename
    
        joblib.dump(best_pipeline,model_path)
        metrics = self.best_model_metrics
        metrics['local_model_path'] = str(model_path)

        metrics_path = model_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f,indent=2)

        # Registring model metrics in all model file
        all_metrics = []
        if os.path.exists(registry_file_path):
            with open(registry_file_path, 'r') as f:
                try:
                    all_metrics = json.load(f)
                    if not isinstance(all_metrics,list):
                        all_metrics = [all_metrics]
                except json.JSONDecodeError:
                    logger.error(f"Error in decoding json file : {registry_file_path}")
        else:
            all_metrics = []

            
        all_metrics.append(metrics)
        with open(registry_file_path, 'w') as f:
            json.dump(all_metrics, f)


        logger.info(f"Model saved to {model_path}")
        return model_path


    def save_reference_profile(self, reference_df: pd.DataFrame) -> Path:
        """Persist training-data statistics used as the drift baseline."""
        profile = {
            "created_at": datetime.now().isoformat(),
            "row_count": int(len(reference_df)),
            "numeric": {
                c: {
                    "mean": float(reference_df[c].mean()),
                    "std": float(reference_df[c].std()),
                    "min": float(reference_df[c].min()),
                    "max": float(reference_df[c].max()),
                    "quantiles": {
                        q: float(reference_df[c].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                    },
                }
                for c in ("MonthlyCharges", "tenure")
                if c in reference_df
            },
            "categorical": {
                c: reference_df[c].astype(str).value_counts(normalize=True).to_dict()
                for c in (
                    "SeniorCitizen", "Partner", "Dependents", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "StreamingMovies", "PaperlessBilling", "PaymentMethod", "Contract",
                )
                if c in reference_df
            },
        }
        path = self.model_dir / "reference_profile.json"
        path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        logger.info(f"Reference profile saved to {path}")
        return path

    def log_drift_metrics(self, reports, multivariate: Dict) -> None:
        """Track drift signals in MLflow so they are visible per model version."""
        with mlflow.start_run(run_name=f"{self.time_stamp}_drift_baseline"):
            for r in reports:
                mlflow.log_metric(f"drift/psi/{r.feature}", r.statistic)
            mlflow.log_metric("drift/multivariate_auc", multivariate.get("auc", float("nan")))

    