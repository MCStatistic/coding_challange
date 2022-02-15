from __future__ import annotations
from ensurepip import bootstrap
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import pandas as pd
import numpy as np
from src.modelling.train_model import ModelTrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path='configurations', config_name='main.yaml')
def run_pipeline(config: DictConfig) -> None:

    logger.info('Configuration File:')
    logger.info(OmegaConf.to_yaml(config))

    logger.info('Data Parameters:')
    for param, value in config.data.items():
        logger.info(f'\t{param} : {value}')

    if config.data.seed == 'None':
        # get a random seed between 0 and 99999 if seed is not defined
        used_seed = np.random.random_integers(low = 0, high = 99999)
    else:
        used_seed = config.data.seed

    # creating data, train model and evaluate
    model = ModelTrainer(training_size=config.data.train_size, 
                         validation_size=config.data.validation_size,
                         test_size=config.data.test_size, 
                         valuerange=tuple([config.data.min_value, config.data.max_value]),
                         seed=used_seed,
                         model=config.train.model,
                         params=config['hyperparameters'])

    mape = model._metrics['mape']
    mse = model._metrics['mse']
    rmse = model._metrics['rmse']
    r2 = model._metrics['r2']
    logger.info(f'Model: {config.train.model}')
    logger.info('Out-Of-Sample Metrics:')
    logger.info(f'\tMAPE: {round(100*mape, 2)}%')
    logger.info(f'\tMSE: {round(mse, 4)}')
    logger.info(f'\tRMSE: {round(rmse, 4)}')

    if config.train.tune:
        model._tune_hyperparameters(k_fold=config.train.k_fold,
                                    n_iter=config.train.n_iter)

    logger.info('Saving results in:')
    logger.info(f'\t{format(os.getcwd())}')

    with open('ModelTrainer.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Saved ModelTrainer as ModelTrainer.pickle')

    with open('model.pickle', 'wb') as handle:
        pickle.dump(model._fit, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Saved fitted model as model.pickle')
    logger.info('Finished Process')


@hydra.main(config_path='configurations', config_name='model_load.yaml')
def load_model(config: DictConfig) -> None:


    logger.info('Configuration File:')
    logger.info(OmegaConf.to_yaml(config))

    base_path = hydra.utils.get_original_cwd()
    logger.info('Base Path:')
    logger.info(f'\t{base_path}')

    # if n_multirun is NOT 'None' the path will be defined accordingly...
    if config.model.n_multirun != 'None':
        model_path = f'{base_path}/multirun/{config.model.date}/{config.model.time}'
        model_path = f'{model_path}/{config.model.n_multirun}'
    # ...otherwise the folder 'outputs' will be used as the iteration of
    # interest was a single run
    else:
        model_path = f'{base_path}/outputs/{config.model.date}/{config.model.time}'

    logger.info('Model Path:')
    logger.info(f'\t{model_path}/model.pickle')

    # reading model.pickle
    model = pd.read_pickle(f'{model_path}/model.pickle')

    logger.info('Loaded model.pickle:')
    logger.info(f'\t{model}')

    # reading ModelTrainer.pickle
    model_class = pd.read_pickle(f'{model_path}/ModelTrainer.pickle')

    # log key information
    logger.info('Loaded ModelTrainer.pickle:')
    logger.info(f'\tSeed for generated data]: {model_class._seed}')
    # get metrics
    mape = model_class._metrics['mape']
    mse = model_class._metrics['mse']
    rmse = model_class._metrics['rmse']
    r2 = model_class._metrics['r2']

    logger.info('Out-Of-Sample Metrics:')
    logger.info(f'\tMAPE: {round(100*mape, 2)}%')
    logger.info(f'\tMSE: {round(mse, 4)}')
    logger.info(f'\tRMSE: {round(rmse, 4)}')
    logger.info(f'\tR^2: {round(r2, 4)}')

    # DO SOMETHING MORE
    # ...
