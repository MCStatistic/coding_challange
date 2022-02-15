import numpy as np

class DataProvider(object):
    def __init__(
            self, training_size: int, validation_size: int,
            test_size: int, valuerange: tuple, 
            seed: int): 
        assert len(valuerange) == 2
        assert valuerange[1] > valuerange[0]

        self._valuerange = valuerange
        self._seed = seed
        self._training = self._generate_data(training_size) 
        self._validation = self._generate_data(validation_size) 
        self._test = self._generate_data(test_size)

    def _generate_data(self, length: int) -> np.ndarray: 
        # set seed
        np.random.seed(self._seed)
    
        return self._valuerange[0] + np.random.rand(length, 2) * (self._valuerange[1] - self._valuerange[0])

    # [] gives a dict-like accessor to permit variable access; # names are 'training', 'validation', 'test', and 'seed'
    def __getitem__(self, field_name: str) -> np.ndarray:
        if field_name == 'training': return self._training
        if field_name == 'validation': return self._validation
        if field_name == 'test': return self._test
        if field_name == 'seed': return self._seed