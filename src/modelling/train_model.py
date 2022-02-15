from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import numpy as np
from src.data.make_dataset import DataProvider
import math
import logging

logger = logging.getLogger(__name__)

class ModelTrainer(DataProvider):

    def __init__(
            self, training_size: int, validation_size: int,
            test_size: int, valuerange: tuple, seed: int, 
            model: str, params: dict):
        super().__init__(training_size, validation_size, 
                         test_size, valuerange, seed)

        logger.info("Random Data Created: ")
        logger.info(f'\tUsed Seed: {self._seed}')
        logger.info(f'\nTraining Data: \n{self._training}\n')
        logger.info(f'\nValidation Data: \n{self._validation}\n')
        logger.info(f'\nTest Data: \n{self._test}\n')

        logger.info('Initialize Model Training')
        # defining possible models
        mod_options = ['LinearReg', 'RandomForest', 'LassoReg']
        # check if specified model is within the options above
        assert model in mod_options, f'{model} is not supported'
        assert model in params.keys(), f'No key for {model} within provided params'

        # adding model type to object
        self._model_type = model

        # creating model object with specified params
        if model == 'LinearReg':
            # check if specified params are available
            available_params = LinearRegression().get_params().keys()
            assert all(x in available_params for x in params[model].keys())

            self._model = LinearRegression(**params[model])
        elif model == 'RandomForest':
            # check if specified params are available
            available_params = RandomForestRegressor().get_params().keys()
            assert all(x in available_params for x in params[model].keys())

            self._model = RandomForestRegressor(**params[model])
        elif model == 'LassoReg':
            # check if specified params are available
            available_params = Lasso().get_params().keys()
            assert all(x in available_params for x in params[model].keys())

            self._model = Lasso(**params[model])

        # fitting model
        self._fit = self._train_model()
        logger.info("Finished Training")
        self._metrics = self._evaluate_model()

    def _train_model(self):
        y_train = self._training[:, 0]
        X_train = self._training[:, 1].reshape(-1, 1)
        fit = self._model.fit(X = X_train, y = y_train)
        # using X_train to get predictions
        y_pred = fit.predict(X_train)
        # logging different error metrics
        # these metrics will not be returned by this function
        # however, it can be calculated afterwards due to available training data
        logger.info('In-Sample Metrics:')
        mse = metrics.mean_squared_error(y_train, y_pred, squared=False)
        rmse = metrics.mean_squared_error(y_train, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_train, y_pred)
        r2 = metrics.r2_score(y_train, y_pred)
        logger.info(f'\tMSE: {round(mse, 4)}')
        logger.info(f'\tRMSE: {round(rmse, 4)}')
        logger.info(f'\tMAPE: {round(100*mape, 2)}%')
        logger.info(f'\tR^2: {round(r2, 4)}')
        # returning fit
        return fit

    def _tune_hyperparameters(self, k_fold: int, n_iter: int):
        # creating tuning grid based on model type
        # could be created within main.py
        # function represent illustrative example of possible usage
        if self._model_type == 'RandomForest':
            n_estimators = [int(x) for x in np.arange(start=100, stop=1000, step=200)]
            bootstrap = [True, False]
            max_depth = [int(x) for x in np.arange(start=10, stop=100, step=10)]
            min_samples_split = [2, 3, 4, 5, 10]
            min_samples_leaf = [1, 2, 3, 4, 5]
            max_features = ['auto', 'sqrt', 'log2']
            tune_grid = {
                'n_estimators': n_estimators,
                'bootstrap': bootstrap,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features
            }

            cv = RandomizedSearchCV(estimator = self._model, param_distributions = tune_grid, 
                                    n_iter = n_iter, cv = k_fold, verbose = 1)

        elif self._model_type == 'LassoReg':
            alpha = [float(x) for x in np.arange(start=0.1, stop=1.0, step=0.05)]
            positive = [True, False]
            max_iter = [int(x) for x in np.arange(start=100, stop=2000, step=200)]
            selection = ['cyclic', 'random']
            tune_grid = {
                'alpha': alpha,
                'max_iter': max_iter,
                'positive': positive,
                'selection': selection
            }
 
            cv = RandomizedSearchCV(estimator = self._model, param_distributions = tune_grid, 
                                    n_iter = n_iter, cv = k_fold, verbose = 1)
        if self._model_type == 'LinearReg':
            logger.info('For LinearReg no tuning is supported')

        else:
            y_train = self._training[:, 0]
            X_train = self._training[:, 1].reshape(-1, 1)
            logger.info('Initialize Randomized Search on Hyperparameters')
            cv_rslt = cv.fit(X_train, y_train)
            # log best params
            logger.info('Best Parameters:')
            for param, value in cv_rslt.best_params_.items():
                logger.info(f'\t{param} : {value}')
            # returning result
            return cv_rslt

    def _evaluate_model(self):
        X_test = self._test[:, 1].reshape(-1, 1)
        y_test = self._test[:, 0]
        # predicting test data
        prediction = self._fit.predict(X = X_test)
        # calculating error metrics without sklearn.metrics
        ape = abs(prediction - y_test) / y_test
        # MAPE
        mape = np.mean(ape)
        # MSE
        mse = np.square((y_test - prediction)).mean()
        # RMSE
        rmse = math.sqrt(mse)
        # r2 by metrics...
        r2 = metrics.r2_score(y_test, prediction)
        # creating dict as output
        out = {
            'mape': mape,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

        return out

    def __getitem__(self, field_name: str) -> np.ndarray:
        if field_name == 'train_data': return self._training
        if field_name == 'val_data': return self._validation
        if field_name == 'test_data': return self._test
        if field_name == 'seed': return self._seed
        if field_name == 'model': return self._model
        if field_name == 'model_type': return self._model_type
        if field_name == 'fit': return self._fit
        if field_name == 'metrics': return self._metrics