import os
from pathlib import Path
from typing import Tuple,List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

random_state=42
test_size=0.20
smote_stratergy=0.1
target_col='Class'


def load_raw(path: str) -> pd.DataFrame:
    logger.info(f'Loading raw data from: {path}')
    df=pd.read_csv(path)

    assert df.shape[1]==31, f'Expected 31 columns, got {df.shape[1]}'
    assert target_col in df.columns,f"'{target_col}' column missing"

    n_fraud=df[target_col].sum()
    n_total=len(df)
    logger.info(f'Loaded {n_total:,}rows|Fraud: {n_fraud} ({n_fraud/n_total*100:.3f}%)')
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Engineering features....')
    df=df.copy()


    df['Amount_log']= np.log1p(df['Amount'])
    df['hour']=(df['Time']//3600)%24
    df['hour_sin']=np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']=np.cos(2*np.pi*df['hour']/24)

    mean_amt=df['Amount'].mean()
    std_amt=df['Amount'].std()
    df['Amount_zscore']=(df['Amount']-mean_amt)/(std_amt+1e-9)
    df['is_high_value']=(df['Amount']>df['Amount'].quantile(0.95)).astype(int)
    df.drop(columns=['Time','Amount','hour'],inplace=True)
    logger.info(f'Feature engineering done -> {df.shape[1]} total columns')
    return df

def build_preprocessor() -> Pipeline:
    return Pipeline(steps=[
        ('scalar',RobustScaler()),
    ])

def split_and_resample(
        df: pd.DataFrame,
        apply_smote: bool=True,        
) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    X=df.drop(columns=[target_col]).values
    y=df[target_col].values

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f'train:{X_train.shape[0]:,}|Test:{X_test.shape[0]:,}')
    logger.info(f"Fraud in train before SMOTE: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")

    if apply_smote:
        smote = SMOTE(
            sampling_strategy=smote_stratergy,
            random_state=random_state,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE → Train: {X_train.shape[0]:,} | Fraud: {y_train.sum():,}")

    return X_train, X_test, y_train, y_test
def load_and_prepare(
    raw_path: str = "data/raw/creditcard.csv",
    output_dir: str = "data/processed",
    apply_smote: bool = True,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, List[str]]:
    """
    Full pipeline: load → engineer → split → scale → resample.

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_preprocessor, feature_names
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load raw CSV
    df = load_raw(raw_path)

    # Engineer features
    df = engineer_features(df)
    feature_names = [c for c in df.columns if c != target_col]

    # Split + SMOTE (on raw features before scaling)
    X_train, X_test, y_train, y_test = split_and_resample(df, apply_smote)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test  = preprocessor.transform(X_test)

    logger.info("Scaling complete — fitted on train only")

    # Save processed arrays to disk
    if save_artifacts:
        np.save(f"{output_dir}/X_train.npy", X_train)
        np.save(f"{output_dir}/X_test.npy",  X_test)
        np.save(f"{output_dir}/y_train.npy", y_train)
        np.save(f"{output_dir}/y_test.npy",  y_test)
        pd.Series(feature_names).to_csv(
            f"{output_dir}/feature_names.csv", index=False
        )
        logger.info(f"Saved processed arrays to {output_dir}/")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, prep, feats = load_and_prepare()

    print("\n✅ Pipeline complete")
    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test shape  : {X_test.shape}")
    print(f"   Fraud in train: {y_train.sum():,}")
    print(f"   Fraud in test : {y_test.sum():,}")
    print(f"   Features      : {feats}")
