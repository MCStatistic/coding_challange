from src.modelling.train_model import ModelTrainer
import pytest

def test_ModelTrainer_model():
    # specified model not within optional models
    with pytest.raises(AssertionError):
        ModelTrainer(1000, 200, 500, tuple([2, 5]), 40, 'GradientBoostingReg', params={})
    # sepcified params not within available params
    with pytest.raises(AssertionError):
        ModelTrainer(1000, 200, 500, tuple([2, 5]), 40, 'LinearReg', params={'LinearReg': {'depth': 10}})
    # empty params
    with pytest.raises(AssertionError):
        ModelTrainer(1000, 200, 500, tuple([2, 5]), 40, 'LinearReg', params = {})