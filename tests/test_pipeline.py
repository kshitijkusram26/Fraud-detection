"""
tests/test_pipeline.py
──────────────────────
Unit tests for data pipeline, predict module and API schemas.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING TESTS
# ════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def _base_df(self, amount=100.0, time=3600.0, label=0):
        import pandas as pd
        return pd.DataFrame({
            **{f"V{i}": [0.0] for i in range(1, 29)},
            "Time":   [time],
            "Amount": [amount],
            "Class":  [label],
        })

    def test_amount_log_created(self):
        from src.data_pipeline import engineer_features
        out = engineer_features(self._base_df(amount=100.0))
        assert "Amount_log" in out.columns
        assert abs(out["Amount_log"].iloc[0] - np.log1p(100.0)) < 1e-9

    def test_cyclical_hour_created(self):
        from src.data_pipeline import engineer_features
        out = engineer_features(self._base_df(time=0.0))
        assert "hour_sin" in out.columns
        assert "hour_cos" in out.columns
        # hour 0 → sin=0, cos=1
        assert abs(out["hour_cos"].iloc[0] - 1.0) < 1e-9

    def test_amount_zscore_created(self):
        from src.data_pipeline import engineer_features
        out = engineer_features(self._base_df())
        assert "Amount_zscore" in out.columns

    def test_high_value_flag(self):
        import pandas as pd
        from src.data_pipeline import engineer_features
        df = pd.DataFrame({
            **{f"V{i}": [0.0, 0.0] for i in range(1, 29)},
            "Time":   [0.0, 0.0],
            "Amount": [1.0, 99999.0],
            "Class":  [0, 1],
        })
        out = engineer_features(df)
        assert "is_high_value" in out.columns
        assert out["is_high_value"].iloc[1] == 1

    def test_raw_time_and_amount_dropped(self):
        from src.data_pipeline import engineer_features
        out = engineer_features(self._base_df())
        assert "Time"   not in out.columns
        assert "Amount" not in out.columns


# ════════════════════════════════════════════════════════════
# PREPROCESSOR TESTS
# ════════════════════════════════════════════════════════════

class TestPreprocessor:

    def test_scaler_output_shape(self):
        from src.data_pipeline import build_preprocessor
        scaler  = build_preprocessor()
        X       = np.random.randn(100, 33)
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == (100, 33)

    def test_scaler_fit_only_on_train(self):
        from src.data_pipeline import build_preprocessor
        scaler  = build_preprocessor()
        X_train = np.random.randn(80, 10)
        X_test  = np.random.randn(20, 10)
        scaler.fit_transform(X_train)
        # Should not raise — uses train statistics
        out = scaler.transform(X_test)
        assert out.shape == (20, 10)


# ════════════════════════════════════════════════════════════
# PREDICTOR TESTS
# ════════════════════════════════════════════════════════════

class TestFraudPredictor:

    @pytest.fixture
    def predictor(self, tmp_path, monkeypatch):
        """Build a tiny dummy model and return a FraudPredictor."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import RobustScaler
        from src.predict import FraudPredictor

        # 33 features = 28 V-features + 5 engineered
        X = np.random.randn(100, 33)
        y = np.zeros(100, dtype=int)
        y[:5] = 1

        model  = LogisticRegression().fit(X, y)
        scaler = RobustScaler().fit(X)

        model_path = tmp_path / "model.pkl"
        prep_path  = tmp_path / "preprocessor.pkl"
        joblib.dump(model,  model_path)
        joblib.dump(scaler, prep_path)

        monkeypatch.setenv("FRAUD_THRESHOLD", "0.5")
        return FraudPredictor(str(model_path), str(prep_path))

    def _txn(self, amount=100.0):
        return {
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": amount,
            "Time":   3600.0,
        }

    def test_predict_one_keys(self, predictor):
        result = predictor.predict_one(self._txn())
        assert "fraud_probability" in result
        assert "prediction"        in result
        assert "Confidence"        in result
        assert result["prediction"]  in ("Fraud", "Legit")
        assert result["Confidence"]  in ("High", "Medium", "Low")
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_predict_batch_length(self, predictor):
        results = predictor.predict_batch([self._txn() for _ in range(5)])
        assert len(results) == 5

    def test_predict_one_no_crash_high_amount(self, predictor):
        result = predictor.predict_one(self._txn(amount=99999.0))
        assert "prediction" in result

    def test_predict_one_zero_amount(self, predictor):
        result = predictor.predict_one(self._txn(amount=0.0))
        assert "fraud_probability" in result


# ════════════════════════════════════════════════════════════
# SCHEMA TESTS
# ════════════════════════════════════════════════════════════

class TestSchemas:

    def _valid(self):
        return {
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": 99.99,
            "Time":   0.0,
        }

    def test_valid_transaction(self):
        from api.schemas import TransactionRequest
        t = TransactionRequest(**self._valid())
        assert t.Amount == 99.99

    def test_negative_amount_rejected(self):
        from pydantic import ValidationError
        from api.schemas import TransactionRequest
        payload = self._valid()
        payload["Amount"] = -10.0
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_missing_v1_rejected(self):
        from pydantic import ValidationError
        from api.schemas import TransactionRequest
        payload = self._valid()
        del payload["V1"]
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_batch_request_valid(self):
        from api.schemas import BatchPredictionRequest, TransactionRequest
        txns = [TransactionRequest(**self._valid()) for _ in range(3)]
        batch = BatchPredictionRequest(transactions=txns)
        assert len(batch.transactions) == 3