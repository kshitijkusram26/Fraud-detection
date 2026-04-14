from typing import Any,Dict,List,Optional
from pydantic import BaseModel,Field,field_validator

class TransactionRequest(BaseModel):
    # PCA features
    V1: float=Field(...,example=-1.3598)
    V2:float=Field(...,example=-0.0728)
    V3:float=Field(...,example=2.5363)
    V4:float=Field(...,example=1.3782)
    V5:  float = Field(0.0)
    V6:  float = Field(0.0)
    V7:  float = Field(0.0)
    V8:  float = Field(0.0)
    V9:  float = Field(0.0)
    V10: float = Field(0.0)
    V11: float = Field(0.0)
    V12: float = Field(0.0)
    V13: float = Field(0.0)
    V14: float = Field(0.0, example=-0.3111)
    V15: float = Field(0.0)
    V16: float = Field(0.0)
    V17: float = Field(0.0, example=-0.5535)
    V18: float = Field(0.0)
    V19: float = Field(0.0)
    V20: float = Field(0.0)
    V21: float = Field(0.0)
    V22: float = Field(0.0)
    V23: float = Field(0.0)
    V24: float = Field(0.0)
    V25: float = Field(0.0)
    V26: float = Field(0.0)
    V27: float = Field(0.0)
    V28: float = Field(0.0)


    Amount:float=Field(...,ge=0.0,example=149.62)
    Time:float=Field(0.0,ge=0.0,example=3600.0)

    @field_validator('Amount')
    @classmethod
    def amount_must_be_positive(cls,v:float)->float:
        if v<0:
            raise ValueError('Amount cannot be negative')
        return v
    
    model_config={
        'json_schema_extra':{
            'example':{
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
                "V4": 1.3782,  "V5": 0.0,     "V6": 0.0,
                "V7": 0.0,     "V8": 0.0,     "V9": 0.0,
                "V10": 0.0,    "V11": 0.0,    "V12": 0.0,
                "V13": 0.0,    "V14": -0.3111,"V15": 0.0,
                "V16": 0.0,    "V17": -0.5535,"V18": 0.0,
                "V19": 0.0,    "V20": 0.0,    "V21": 0.0,
                "V22": 0.0,    "V23": 0.0,    "V24": 0.0,
                "V25": 0.0,    "V26": 0.0,    "V27": 0.0,
                "V28": 0.0,    "Amount": 149.62, "Time": 3600.0,
            }
        }
    }

class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]=Field(
        ...,min_length=1,max_length=5000
    )


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability:float=Field(...,description='Score between 0.0 and 1.0')
    prediction:str=Field(...,description='Fraud or Legit')
    confidence:str=Field(...,description='High/Medium/Low')
    model_version:str
    latency_ms:float

class BatchPredictionResponse(BaseModel):
    total: int
    fraud_count: int
    legit_count: int
    fraud_rate: float
    predictions: List[Dict[str,Any]]
    latency_ms:float
    model_version:str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    api_version: str

class ModelInfoResponse(BaseModel):
    version: str
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    f1_score:Optional[float]
    threshold: float
    notes: Optional[str]

class MetricsResponse(BaseModel):
    session_total: int
    session_fraud: int
    session_legit: int
    db_total: int
    db_fraud:int
    fraud_rate_db:float
