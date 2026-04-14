import os
from typing import Dict,Any,List
import joblib
import numpy as np
from dotenv import load_dotenv
from loguru import logger
load_dotenv()

FEATURE_ORDER=(
    [f'V{i}'for i in range(1,29)]
    + ['Amount_log','hour_sin','hour_cos','Amount_Zscore','is_high_value']
)

dataset_mean_amount=88.35
dataset_std_amount=250.12


class FraudPredictor:
    def __init__(
            self,
            model_path: str=None,
            preprocessor_path: str=None,
    ):
        model_path        = model_path        or os.getenv("MODEL_PATH",        "models/best_model.pkl")
        preprocessor_path = preprocessor_path or os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
        logger.info(f'Loading model from: {model_path}')
        logger.info(f'Loading preprocessor from: {preprocessor_path}')


        self.model=joblib.load(model_path)
        self.preprocessor=joblib.load(preprocessor_path)
        self.threshold=float(os.getenv('Fraud_threshold','0.5'))

        logger.info('FraudPredictor ready')
        
    

    def _engineer(self,raw:Dict[str,Any])->Dict[str,float]:
        d=dict(raw)

        amount =float(d.get('Amount',0))
        time_s=float(d.get('Time',0))
        d['Amount_log']=float(np.log1p(amount))
        hour=(time_s//3600)%24
        d['hour_sin']=float(np.sin(2*np.pi*hour/24))
        d['hour_cos']=float(np.cos(2*np.pi*hour/24))
        d['Amount_zscore']=(amount-dataset_mean_amount)/dataset_std_amount
        d['is_high_value']=int(amount>220)

        return d
    
    def _to_array(self,raw:Dict[str,Any])->np.ndarray:
        d=self._engineer(raw)
        row=np.array(
            [[d.get(f,0.0)for f in FEATURE_ORDER]],
            dtype=np.float64
        )
        return self.preprocessor.transform(row)
    

    #Confidence Level
    @staticmethod
    def _confidence(prob:float)->str:
        if prob>=0.8:
            return 'High'
        elif prob>=0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def predict_one(self,raw:Dict[str,Any])->Dict[str,Any]:
        X=self._to_array(raw)
        prob=float(self.model.predict_proba(X)[0,1])

        return {
        "fraud_probability": round(prob, 5),
        "prediction":        "FRAUD" if prob >= self.threshold else "LEGIT",
        "confidence":        self._confidence(prob),
        "threshold_used":    self.threshold,
    }
    
    def predict_batch(
            self,record:List[Dict[str,Any]]
    )->List[Dict[str,Any]]:
        rows=np.vstack([self._to_array(r) for r in record])
        probs=self.model.predict_proba(rows)[:,1]


        return[
            {
            'fraud_probability': round(float(p),5),
            'predictions':'fruad' if p>=self.threshold else "Legit",
            'confidence':self._confidence(float(p)),
            'threshold_used':self.threshold
            }
            for p in probs
        ]
    
_predictor:FraudPredictor=None

def get_predictor()->FraudPredictor:
    global _predictor
    if _predictor is None:
        _predictor =FraudPredictor()
    return _predictor

    
    