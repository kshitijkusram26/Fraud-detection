import argparse 
import  os 
import time
import warnings
from pathlib import Path
from typing import Dict,Any
import joblib 
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier
from src.data_pipeline import load_and_prepare
warnings.filterwarnings('ignore')
load_dotenv()

mlflow_uri=os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_experiment("fraud-detection")
experiment='fraud-detection'
model_dir=Path('D:\\PROJECTS\\fraud-detection\\models')
random_state=42

model_dir.mkdir(exist_ok=True)

model_configs: Dict[str, Any]={
    'logistic regression':{
        'model': LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
        ),
        'params':{
            'C':0.1,
            'class_weight':'balanced',
            'solver':'lbfgs',
        },
    },
    'random_forest':{
        'model': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        ),
        'params':{
            'n_estimators':200,
            'max_depth':12,
            'min_samples_leaf':4,
            'class_weight':'balanced'
        },
    },
    'xgboost':{
        'model':XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weights=577,
            eval_metrics='aucpr',
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        'params':{
            'n_estimators':300,
            'max_depth':6,
            'learning_rate':0.05,
            'scale_pos_weight':577
        },
    },
    'lightgbm':{
        'model':None,
        'params':{
            'n_estimators':300,
            'num_leaves':63,
            'learning_rate':0.05,
            'is_unbalance':True
        },
    },
}



def find_best_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
)-> float:
    best_f1,best_thresh=0.0,0.5
    for t in np.arange(0.1,0.9,0.02):
        y_pred=(y_prob>=t).astype(int)
        f1=f1_score(y_true,y_pred,zero_division=0)
        if f1>best_f1:
            best_f1,best_thresh=f1,float(t)
    return round(best_thresh,2)


def evaluate(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float=0.5,
)-> Dict[str,float]:
    y_prob=model.predict_proba(X_test)[:,1]
    y_pred=(y_prob>=threshold).astype(int)

    return {
        'pr_auc': round(average_precision_score(y_test,y_prob),5),
        'roc_auc': round(roc_auc_score(y_test,y_prob),5),
        'f1': round(f1_score(y_test,y_pred,zero_division=0),5),
        'precision': round(precision_score(y_test,y_pred,zero_division=0),5),
        'recall': round(recall_score(y_test,y_pred,zero_division=0),5),
        'threshold':threshold,
    }



def train_model(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> Dict[str,Any]:
    cfg   = model_configs[model_name]
    model = cfg["model"]

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=f"{model_name}") as run:
        logger.info(f'\n{'='*50}\nTraining: {model_name}\n{'='*50}')
        
        mlflow.log_params(cfg['params'])
        mlflow.log_param('model_type',  model_name)
        mlflow.log_param('training_samples',X_train.shape[0])
        mlflow.log_param('n_features',X_train.shape[1])

        t0=time.time()
        model.fit(X_train,y_train)
        train_time=round(time.time()-t0,2)

        y_prob=model.predict_proba(X_test)[:,1]
        best_thresh=find_best_threshold(y_test,y_prob)



        metrics=evaluate(model,X_test,y_test,threshold=best_thresh)
        metrics['train_time_sec']=train_time

        mlflow.log_metrics(metrics)


        logger.info(f'PR-AUC: {metrics['pr_auc']:.4f}')
        logger.info(f'ROC-AUC :{metrics['roc_auc']:.4f}')
        logger.info(f'F1: {metrics['f1']:.4f}')
        logger.info(f'Precision: {metrics['precision']:.4f}')
        logger.info(f'Recall: {metrics['recall']:.4f}')
        logger.info(f'Threshold: {best_thresh}')
        logger.info(f'Train Time: {train_time}s')


        if hasattr(model, 'feature_importance'):
            fi_df=pd.DataFrame({
                'feature':feature_names,
                'importance':model.feature_importances_,
            }).sort_values('importance',ascending=False)

            fi_path=f'models/features_importances_{model_name}.csv'
            fi_df.to_csv(fi_path,index=False)
            mlflow.log_artifact(fi_path)
        
        model_path=model_dir/f'{model_name}.pkl'
        joblib.dump(model,model_path)
        mlflow.log_artifact(str(model_path))


        run_id=run.info.run_id
        logger.info(f'MLFLOW run_id: {run_id}')


    return{
        'model':model,
        'metrics':metrics,
        'run_id':run_id,
        'name':model_name,
    }


def main(model_filter: str='all',apply_smote: bool=True):
    X_train,X_test,y_train,y_test,preprocessor,feature_names=load_and_prepare(
        apply_smote=apply_smote
    )

    joblib.dump(preprocessor,model_dir/'preprocessor.pkl')
    logger.info('saved preprocessor.pkl')

    if model_filter=='all':
         models_to_train=list(model_configs.keys())
    else:
        models_to_train=[model_filter]
    results = []
    for name in models_to_train:
        try:
            result = train_model(
                name, X_train, X_test, y_train, y_test, feature_names
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            import traceback
            traceback.print_exc()
    

    best=max(results,key=lambda r:r['metrics']['pr_auc'])
    best_path=model_dir/'best_model.pkl'
    joblib.dump(best['model'],best_path)
    logger.info(f'\nBest Model:{best['name']}')
    logger.info(f'PR_AUC: {best['metrics']['pr_auc']:.4f}')
    logger.info(f'Saved to: {best_path}')

    print('\n----Model Comparison-----')
    print(f'{'model':<25}{'PR-AUC':>8}{'ROC-AUC':>9}{'F1':>8}{'Threshold':>10}')
    print("-"*65)
    for r in sorted(results,key=lambda x: x['metrics']['pr_auc'],reverse=True):
        m=r['metrics']
        print(
            f'{r['name']:<25}{m['pr_auc']:>8.4f}'
            f'{m['roc_auc']:>9.4f}{m['f1']:>8.4f}'
            f'{m['threshold']:>10.2f}'
        )

    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='all',
        choices=['all','logistic_regression','random_forest','xgboost','lightgbm'],
    )
    parser.add_argument('--no-smote',action='store_true')
    args=parser.parse_args()

    main(model_filter=args.model,apply_smote=not args.no_smote)




