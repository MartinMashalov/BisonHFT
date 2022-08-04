"""run the full script"""
import time
import schedule
from script_facade import main
from script_tuner import run_nni
from bison_create import create_model
from json import load
from fetch_alpaca_data import fetch_action

# set schedule for signal called
schedule.every().day.at('09:35').do(main)

# get new data for the day
schedule.every().day.at('16:30').do(fetch_action)

# set schedule for tuner and re-trainer
schedule.every().day.at('17:00').do(run_nni, 'base')

# set schedule for final training of the model
with open('model_configs.json', 'r') as config_file:
    json_configs = load(config_file)
    years = json_configs['years']
    window, days = json_configs['window'], json_configs['days']
    wavelet = json_configs['wavelet']
