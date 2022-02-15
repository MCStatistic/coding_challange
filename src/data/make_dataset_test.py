from src.data.make_dataset import DataProvider
import pytest
import numpy as np

def test_DataProvider():
    with pytest.raises(ValueError):
        DataProvider(-1, 0, 3, tuple([-1, 0]), 40)
    with pytest.raises(ValueError):
        DataProvider(10, -100, 5, tuple([-1, 0]), 40)
    with pytest.raises(AssertionError):
        DataProvider(10, 100, 5, tuple([10, -20]), 40)

def test_DataProvider_shape():
    data = DataProvider(training_size=100, validation_size=20, test_size=30,
                        valuerange=tuple([2, 5]), seed = 2)
    assert data['training'].shape == (100, 2)
    assert data['validation'].shape == (20, 2)
    assert data['test'].shape == (30, 2)

def test_DataProvider_range():
    data = DataProvider(training_size=50000, validation_size=10000, test_size=20000,
                        valuerange=tuple([2, 5]), seed = 42)

    for values in (data['training'], data['validation'], data['test']):
        assert round(values.min(), 4) == 2.0
        assert round(values.max(), 4) == 5.0