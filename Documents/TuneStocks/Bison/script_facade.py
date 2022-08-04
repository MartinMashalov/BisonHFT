import time
from twilio.rest import Client
import pywt
import logging
logging.basicConfig()
logger = logging.getLogger('my-logger')
logger.setLevel(logging.ERROR)
from sklearn.preprocessing import quantile_transform
import yfinance as yf
from datetime import datetime
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import pandas_ta as pd_ta
import numpy as np
from bison_create import create_model
from fetch_alpaca_data import fetch_alpaca_csv
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, roc_auc_score
from datetime import timedelta
import pandas_market_calendars as mcal
from pydantic import BaseModel
from typing import Any
from json import load
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# set numpy random seed
np.random.RandomState(42)
np.random.seed(42)

# create twilio client message service
class ScriptVars(BaseModel):
    client: Any = Client("AC0d79e56293d4494c36eee4f48a59ff8e", 'f1e24705d7289d4b6d680c8c04beb484')
    version: str = '1.06'
    confidence_threshold: float = 0.9
    symbol: str = 'SPY'
    interval: str = '30m'
    lookback: int = 14
    shuffle_split: bool = True
    bot_phone_num: str = "+15706725011"
    bot_phone_contacts: list = ["+19142822807", "+19145126792"]

# create instance of base model for configurations on general Bison level
general_configs = ScriptVars()

# set up running variables
with open('model_configs.json', 'r') as config_file:
    json_configs = load(config_file)
    param_sets = json_configs['model_params']

shuffle_split: bool = True

# training data boundaries test to avoid extrapolation issues
training_data_boundaries = lambda min, max, val: True if val < min or val > max else False

def test_boundaries_whole(df, ranges, training_data_boundaries):
    boundary_flag: bool = True
    for var, range in zip(['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k'], ranges):
        if training_data_boundaries(range[0], range[1], df[var]):
            boundary_flag = False
    return boundary_flag

def create_model_local(years=None, trend=None, wavelet=None):
    # get data from backend
    data = fetch_alpaca_csv(years)
    data.rename(columns={col: col.lower() for col in data.columns}, inplace=True)
    copy_data = data.copy()

    print(data.shape)  # replace with wavelet transformed version
    data['open'], data['high'], data['low'], data['close'], data['volume'] = _wavelet_smooth(data, wavelet)

    # create indicator features
    data = _get_indicator_data(data)
    copy_data = _get_indicator_data(copy_data)
    data.drop('unnamed: 0', axis=1, inplace=True)
    copy_data.drop('unnamed: 0', axis=1, inplace=True)
    if data.isna().sum().sum() != copy_data.isna().sum().sum():
        print('NaN Error')
        return
    data.dropna(inplace=True)
    copy_data.dropna(inplace=True)

    # remove nan values and reset indices on both dataframes
    data.reset_index(inplace=True)
    copy_data.reset_index(inplace=True)
    data.drop(['index'], axis=1, inplace=True)
    copy_data.drop(['index'], axis=1, inplace=True)
    data['pred'] = _produce_prediction(copy_data, window=trend)

    # create live data samples from equivalent training distribution
    live_data = data[-trend:].drop('pred', axis=1)[['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k']]
    strike_prices = copy_data[-trend:]['close']
    live_dates = data[-trend:]['timestamp']

    # resolve formatting and nan issues
    del (data['close'])
    data.dropna(inplace=True)
    print(data.shape, 'tracking data shape 4')
    data = data.sample(frac=1).reset_index(drop=True)
    features = ['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k']
    target = 'pred'

    # train-test-validation data splits
    test_ratio = 0.20
    X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                        data[target], test_size=test_ratio, random_state=42)

    # get ranges of feature variables to check for out-of-sample predictions
    rsi_range = [X_train['rsi'].min(), X_train['rsi'].max()]
    macd_range = [X_train['macd'].min(), X_train['macd'].max()]
    willr_range = [X_train['willr'].min(), X_train['willr'].max()]
    obv_range = [X_train['obv'].min(), X_train['obv'].max()]
    proc_range = [X_train['proc'].min(), X_train['proc'].max()]
    stoch_k = [X_train['stoch_k'].min(), X_train['stoch_k'].max()]
    ranges = [rsi_range, macd_range, willr_range, obv_range, proc_range, stoch_k]
    ratio = Counter(y_train)
    scale_pos_weight = max(ratio.values()) / min(ratio.values())
    print(scale_pos_weight)

    # transform distributions into the normal distribution to standardize training-testing-live samples
    live_data = live_data[features]
    output_dist = 'uniform'
    train_trans = quantile_transform(X_train, output_distribution=output_dist, n_quantiles=len(X_train))
    test_trans = quantile_transform(X_test, output_distribution=output_dist, n_quantiles=len(X_train))
    live_trans = quantile_transform(live_data, output_distribution=output_dist, n_quantiles=len(X_train))
    for i, col in enumerate(X_train.columns):
        X_train[col] = train_trans[:, i]
        X_test[col] = test_trans[:, i]
        live_data[col] = live_trans[:, i]

    """# transform further into Gaussian distribution if samples are still significantly different
    train_trans = power_transform(X_train, method='yeo-johnson')
    test_trans = power_transform(X_test, method='yeo-johnson')
    live_trans = power_transform(live_data, method='yeo-johnson')
    for i, col in enumerate(X_train.columns):
        X_train[col] = train_trans[:, i]
        X_test[col] = test_trans[:, i]
        live_data[col] = live_trans[:, i]

    # final Gaussian transform across all samples
    train_trans = power_transform(X_train, method='yeo-johnson')
    test_trans = power_transform(X_test, method='yeo-johnson')
    live_trans = power_transform(live_data, method='yeo-johnson')
    for i, col in enumerate(X_train.columns):
        X_train[col] = train_trans[:, i]
        X_test[col] = test_trans[:, i]
        live_data[col] = live_trans[:, i]"""

    # create and train xgboost decision tree model
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, seed=42).fit(X_train, y_train)

    # p_value_ks = eval_drift(model, X_train, live_data[features])
    stat, p_value = ttest_ind(X_train, live_data)
    print("T-test P-value: ", p_value)

    # determine accuracy and append to results
    rf_prediction = model.predict(X_test)
    print("Raw Accuracy: ", accuracy_score(list(y_test), list(rf_prediction)))
    print("ROC AUC: ", roc_auc_score(y_test, rf_prediction))
    print('F1 Score: ', f1_score(y_test, rf_prediction, average='macro'))
    rf_accuracy = balanced_accuracy_score(list(y_test), list(rf_prediction))

    # do cross validation test
    #cross_val_acc = cross_val_score(model, data[features], data[target], cv=5, scoring='f1_macro').mean()

    # make live predictions
    live_predictions = model.predict(live_data)
    live_prob_predictions = model.predict_proba(live_data)
    live_data['pred'] = live_predictions
    live_data['strikes'] = strike_prices
    live_data['prob'] = [i[np.argmax(i)] for i in live_prob_predictions]
    live_data['date'] = live_dates
    live_data['window'] = [trend // 14 for _ in range(live_data.shape[0])]
    live_data['boundary'] = [test_boundaries_whole(live_data[['rsi', 'macd',
                                                              'willr', 'obv', 'proc', 'stoch_k']].iloc[i],
                                                   ranges, training_data_boundaries)
                             for i in range(live_data.shape[0])]
    live_data.rename(columns={'pred': 'trend', 'date': 'start_date'}, inplace=True)

    # calculate accuracy score on live data
    print('Test Accuracy: ', rf_accuracy)

    return model, rf_accuracy, live_data[['trend', 'prob', 'strikes', 'start_date']]

# send sms through twilio API
def send_sms(message: str):
    """send sms message with stock update"""
    for number in general_configs.bot_phone_contacts:
        general_configs.client.messages.create(to=number, from_=general_configs.bot_phone_num, body=message)

# preprocessing and de-noising functions
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# de-noise the signal with wavelet function
def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

# smooth the initial features with wavelets to create simpler curve for learning
def _wavelet_smooth(data, wavelet):
    """wavelet transformation applied to curve"""

    processed_arr = []

    for col in ['open', 'high', 'low', 'close', 'volume']:
        processed_arr.append(wavelet_denoising(data[col], wavelet=wavelet, level=1)[:len(data)])
    return processed_arr

# create indicator features
def _get_indicator_data(data):
    # technical analysis indicators
    data['rsi'] = pd_ta.rsi(data['close'])
    data['macd'] = pd_ta.macd(data['close']).iloc[:, -1]
    data['willr'] = pd_ta.willr(data['high'], data['low'], data['close'])
    data['obv'] = pd_ta.obv(data['close'], data['volume'])
    data['proc'] = pd_ta.roc(data['close'])
    data['stoch_k'] = pd_ta.stoch(data['high'], data['low'], data['close']).iloc[:, 0]

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    try:
        del (data['Adj Close'])
    except KeyError:
        pass
    return data

# create target variables
def _produce_prediction(data, window):
    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    return data['pred']

# get closing date of the trade based on trading window size, avoiding weekends and holidays
def get_markets_days_ahead(start_date, days):
    date_1 = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date = date_1 + timedelta(days=days)
    # Create a calendar
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date=str(date_1), end_date=str(end_date))
    early_markets_days = early.shape[0]
    print(early_markets_days)

    # check if end date is already a market day
    if early_markets_days -1 == days:
        return str(end_date).split(' ')[0]

    n = days
    while early_markets_days != days:
        end_date = date_1 + timedelta(days=n+1)
        n += 1
        print(start_date, end_date)
        early = nyse.schedule(start_date=str(date_1), end_date=str(end_date))
        early_markets_days = early.shape[0]

    return str(early.iloc[-1]['market_open']).split(' ')[0]

# core runner function to pull features and preprocess
def run(years=None, wavelet=None):
    # pull data from backend
    df = pd.read_csv(f'alpaca_data_yearly/{years}_years_alpaca.csv').drop('Unnamed: 0', axis=1)
    new_data = yf.download(general_configs.symbol,
                           start=datetime.today().strftime('%Y-%m-%d'),
                           end=datetime.now(),
                           interval=general_configs.interval)[:-1] # remove the dynamic last input (not closed yet)
    new_data['Timestamp'] = new_data.index
    new_data.reset_index(inplace=True, drop=True)
    new_data.drop(['Adj Close'], 1, inplace=True)
    new_data = new_data[['Timestamp', 'Open', "High", 'Low', 'Close', 'Volume']]
    new_data['Timestamp'] = new_data['Timestamp'].apply(lambda x: str(x).replace('-04:00', ''))
    new_data['time'] = new_data['Timestamp'].apply(lambda x: x.split(' ')[1])
    new_data['day'] = new_data['Timestamp'].apply(lambda x: x.split(' ')[0])
    data = df.append(new_data)
    print(new_data)
    data.rename(columns={col: col.lower() for col in data.columns}, inplace=True)
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)

    # de-noise data
    try:
        data['open'], data['high'], data['low'], data['close'], data['volume'] = _wavelet_smooth(data, wavelet)
    except:
        raise BrokenPipeError

    # create indicator features
    data = _get_indicator_data(data)
    data.dropna(inplace=True)

    X = data[data['day'] == datetime.today().strftime('%Y-%m-%d')][['rsi', 'macd',
                                                    'willr', 'obv', 'proc', 'stoch_k']].iloc[-general_configs.lookback:]
    timestamps = data[data['day'] == datetime.today().strftime('%Y-%m-%d')]['timestamp']
    return X, timestamps

# get current strike price
current_strike = yf.download('SPY', interval='1m', period='1d')['Close'][-2]
print('Current Strike Price: ', current_strike)
lower_strike, higher_strike = current_strike-3, current_strike+3

def main():
    """main runner function"""
    # main program to run prediction and sms sent
    send_messages: bool = False
    i1 = 1
    i2 = 2
    threshold_95 = False
    for param_set in param_sets[i1:i2]:
        # import BisonML model
        print(param_set)
        days = param_set['days']
        model, accuracy, live_preds = create_model(param_set['years'], param_set['window'], param_set['wavelet'],
                                                    facade=True)
        # upgrade threshold for lower accuracy models
        if accuracy < 0.8 or threshold_95:
            print('in threshold change')
            general_configs.confidence_threshold = 0.95

        # change the threshold if presented with too many high confidence trades
        live_predictions = live_preds[live_preds['prob'] > general_configs.confidence_threshold]
        # preprocess trading table
        message_table: dict = {0: "DOWN", 1: "UP"}
        live_predictions['trend'] = live_predictions['trend'].apply(lambda x: message_table[x])
        live_predictions['trading_window'] = [days for _ in range(live_predictions.shape[0])]
        # filter for predictions close to the strike price
        live_predictions = live_predictions[(live_predictions['strikes'] > lower_strike) &
                                            (live_predictions['strikes'] < higher_strike)]
        live_predictions.reset_index(inplace=True)
        live_predictions['day'] = live_predictions['start_date'].apply(lambda x: str(x).split(' ')[0])
        live_predictions = live_predictions[live_predictions['day'].isin(live_predictions['day'].unique()[-1:]
                                                                         )].drop('day', axis=1)
        live_predictions.to_csv("live_preds_test.csv")
        print(live_predictions[['trading_window', 'trend', 'prob', 'strikes', 'start_date']])

        # formulate message
        message_body: str = f"""\n 
    -----------------------
    MODEL ACC SUMMARY: {round(100*accuracy, 2)}% \nTRADING WINDOW: {days}D 
    ----------------------- \n 
                          """
        if send_messages:
            send_sms(message_body)
        for index, row in live_predictions.iterrows():
            sub_message = f"""
    Trend: {row['trend']}
    Window: {row['trading_window']}D
    Date: {row['start_date']} 
    Strike: ${row['strikes']} 
    Conf: {round(100*row['prob'], 1)}% 
    ----------------------- \n 
                          """
            if send_messages:
                send_sms(sub_message)
        # save prediction table for future reference
        live_predictions.to_csv(f"intraday_model_preds/{live_predictions['start_date'].iloc[-1]}_pred_{days}D.csv")

main()