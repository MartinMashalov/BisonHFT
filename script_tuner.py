"""wavelet transformation tuning Bison"""
from pathlib import Path
from pydantic import BaseModel
from nni.experiment import Experiment
import warnings
warnings.filterwarnings('ignore')
import pywt
from typing import Any
from time import sleep
from json import dump

# define non-discrete wavelets
null_wavelets: list = ["cgau1", "cgau2", "cgau3", "cgau4", "cgau5", "cgau6", "cgau7", "cgau8", "cmor", "fbsp",
                "gaus1", "gaus2", "gaus3", "gaus4", "gaus5", "gaus6", "gaus7", "gaus8", "mexh", "morl", "shan"]

class TuningConfigs(BaseModel):
    """tuning configuration variables for NNI tuner"""
    internal_search_space_estimators: dict = {
        'max_depth': {'_type': 'choice', '_value': [4, 5, 6, 7]},
        'n_estimators': {'_type': 'choice', '_value': [80, 100, 150, 200, 250,300]},
        'learning_rate': {'_type': 'choice', '_value': [0.1, 0.3, 0.5, 0.05]}
    }
    internal_search_space_regularizers: dict = {
        'reg_alpha': {'_type': 'uniform', '_value': [0, 1]},
        'reg_lambda': {'_type': 'uniform', '_value': [0, 1]}
    }
    deterministic_search_space: dict = {
        'subsample': {'_type': 'uniform', '_value': [0, 1]},
        'colsample_bytree': {'_type': 'uniform', '_value': [0, 1]},
        'colsample_bylevel': {'_type': 'uniform', '_value': [0, 1]},
    }
    search_space: dict = {
        'years': {'_type': 'choice', '_value': range(3, 18)},
        'window': {'_type': 'choice', '_value': [14, 28, 42, 56, 70, 84, 98]},
        'wavelet': {'_type': 'choice', '_value': [wave for wave in pywt.wavelist() if wave not in null_wavelets]}
    }
    tuner: str = 'Random'
    max_trials: int = 2000
    concurrent_trials: int = 6
    max_duration: str = '2h'
    web_host: str = 'local'

# initialize variable for configuration base model
tuner_configs = TuningConfigs()

def organize_trials(exported_trial: Any, max_models=3) -> list:
    """organize, sort, and retrieve best trials from all results"""

    # create object to hold processed trial data
    trial_container: list = []

    # synthesize trial data
    for trial in exported_trial:
        try:
            trial_container.append((float(trial.value), trial.parameter))
        except TypeError:
            continue
    trial_container.sort(key=lambda x: x[0])

    # retrieve top 5 best parameters with highest metric
    for i in trial_container:
        print(i[1], i[0])
    best_metrics: list = [i[1] for i in trial_container[-max_models:]]
    return best_metrics

def parameter_facade(experiment, file_name: str) -> None:
    """facade function for all parameter functions"""

    # sort exported trial data
    best_params: list = organize_trials(experiment.export_data())
    json_container: list = []
    for param_set in best_params:
        param_set['days'] = param_set['window'] // 14
        json_container.append(param_set)

    # store in a configuration file
    with open(file_name, 'w') as config_file:
        dump({'model_params': best_params}, config_file, indent=' ')

def run_nni(tuning_type: str, default_name: str = 'FreeRangeBison'):
    """run nni tuner main function"""
    # Configure experiment
    experiment = Experiment(tuner_configs.web_host)
    if tuning_type == 'base' or tuning_type == 'est':
        experiment.config.trial_command = 'python3 new_bison_create.py'
    else:
        experiment.config.trial_command = 'python3 bison_sampling.py'
    experiment.config.trial_code_directory = Path(__file__).parent

    # set search space for parameter tuning
    if tuning_type == 'base':
        experiment.config.search_space = tuner_configs.search_space
    elif tuning_type == 'est':
        experiment.config.search_space = tuner_configs.internal_search_space_estimators
    elif tuning_type == 'reg':
        experiment.config.search_space = tuner_configs.internal_search_space_regularizers
    elif tuning_type == 'det':
        experiment.config.search_space = tuner_configs.deterministic_search_space

    # additional parameters
    experiment.config.tuner.name = tuner_configs.tuner
    experiment.config.experiment_name = default_name
    experiment.config.max_trial_number = tuner_configs.max_trials
    experiment.config.trial_concurrency = tuner_configs.concurrent_trials
    #experiment.config.max_experiment_duration = tuner_configs.max_duration

    # set the file names beforehand
    if tuning_type == 'base':
        file_name = 'model_configs.json'
    elif tuning_type == 'det':
        file_name = 'det_configs.json'
    else:
        file_name = 'general_configs.json'

    # run NNI experiment
    try:
        experiment.run(port=8000, wait_completion=True)
    except AssertionError:
        parameter_facade(experiment, file_name)
        return
    parameter_facade(experiment, file_name)
    # cut off the web host
    sleep(10)
    experiment.stop()

run_nni('base')