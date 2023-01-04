import pandas as pd
import nni
import pandas_ta as pd_ta
from hep_ml import reweight
import matplotlib.pyplot as plt
from hep_ml.metrics_utils import ks_2samp_weighted
from fetch_alpaca_data import fetch_alpaca_csv
from pandas import DataFrame
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score
import warnings
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import pywt
import pandas_market_calendars as mcal
from collections import Counter
import numpy as np
from time import perf_counter
import yfinance as yf
from scipy.stats import ttest_ind
from sklearn.preprocessing import quantile_transform

warnings.filterwarnings('ignore')

# set random seed on numpy
np.random.RandomState(42)
np.random.seed(42)

# performance csv headers
header = ['test_acc', 'live_acc']

params: dict = {
    'years': 3,
    'wavelet': 'bior1.1',
    'window': 70
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

# create script parameters (dynamic)
years = params['years']
trend = params['window']
wavelet = params['wavelet']
shuffle_split: bool = True

# find yesterday closing price
ticker = yf.Ticker('SPY')
yesterday_price = ticker.history(interval='1d')['Close'][-2]

# mean absolute error evaluator for wavelet transform
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# de-noise original signal with selected wavelet function
def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

# smooth initial model features in order to extract longer trend pattern
def _wavelet_smooth(data, wavelet):
    """wavelet transformation applied to curve"""
    processed_arr = []
    for col in ['open', 'high', 'low', 'close', 'volume']:
        processed_arr.append(wavelet_denoising(data[col], wavelet=wavelet, level=1))
    return processed_arr

def _get_indicator_data(data_raw):
    # technical analysis indicators
    data_raw['rsi'] = pd_ta.rsi(data_raw['close'])
    data_raw['macd'] = pd_ta.macd(data_raw['close']).iloc[:, -1]
    data_raw['willr'] = pd_ta.willr(data_raw['high'], data_raw['low'], data_raw['close'])
    data_raw['obv'] = pd_ta.obv(data_raw['close'], data_raw['volume'])
    data_raw['proc'] = pd_ta.roc(data_raw['close'])
    data_raw['stoch_k'] = pd_ta.stoch(data_raw['high'], data_raw['low'], data_raw['close']).iloc[:, 0]

    # Remove columns that won't be used as features
    del (data_raw['open'])
    del (data_raw['high'])
    del (data_raw['low'])
    del (data_raw['volume'])
    try:
        del (data_raw['Adj Close'])
    except KeyError:
        pass
    return data_raw

def _produce_prediction_gaps(data, transformed_data, window, from_hardcoded=False, year=None) -> DataFrame:
    """create predicted target values"""
    try:
        if from_hardcoded:
            return pd.read_csv(f'targets/{year}y_{window}w.csv')
    except FileNotFoundError:
        pass
    # create containers and calendars
    nyse = mcal.get_calendar('NYSE')
    calendar = nyse.schedule(start_date=data['day'].iloc[0], end_date=data['day'].iloc[-1])
    pred_days = data['day'].unique()[-(window // 14):]
    dropped_idx: list = []
    missing_days = []
    missing_counter = 0
    value_counts = data['day'].value_counts()
    for key, value in value_counts.items():
        if value != 14:
            missing_days.append(key)
            missing_counter += 1
            print(value, key)
    data = data.loc[~data['day'].isin(missing_days)]
    transformed_data = transformed_data.loc[~transformed_data['day'].isin(missing_days)]

    # custom mapping for trading window at given intraday time
    map: dict = {
        '09:30:00': 13, '10:00:00': 12, '10:30:00': 11, '11:00:00': 10, '11:30:00': 9, "12:00:00": 8,
        '12:30:00': 7, '13:00:00': 6, '13:30:00': 5, '14:00:00': 4, '14:30:00': 3, '15:00:00': 2,
        '15:30:00': 1, '16:00:00': 0
    }
    # create containers
    times, closes, days = data['time'], data['close'], data['day']
    targets = []
    end_dates = []
    compare_closes = []

    for start_idx in tqdm(range(data.shape[0])):
        # get prices to compare for target variable
        current_close = closes.iloc[start_idx]

        if days.iloc[start_idx] in pred_days:
            end_dates.append(np.nan)
            compare_closes.append(np.nan)
            targets.append(np.nan)
            continue

        # get price comparisons
        try:
            compare_close = closes.iloc[start_idx + window + map[times.iloc[start_idx]]]
            end_dates.append(days.iloc[start_idx + window + map[times.iloc[start_idx]]])
            compare_closes.append(compare_close)
        except IndexError:
            compare_close = np.nan
            end_dates.append(np.nan)
            compare_closes.append(np.nan)  # only at the -window span at the end of the dataframe

        # check market day difference
        if not np.isnan(compare_close):
            diff = calendar.loc[days.iloc[start_idx]:days.iloc[start_idx + window + map[times.iloc[start_idx]]]].shape[
                       0] - 1
            if diff != window // 14 or '16:00:00' not in times.iloc[
                start_idx + window + map[times.iloc[start_idx]]] or compare_close == current_close:
                dropped_idx.append(start_idx)
                targets.append('GAP')
                continue
            else:
                if current_close < compare_close:
                    targets.append(1)
                else:
                    targets.append(0)
        else:
            # check if larger or smaller
            targets.append('GAP')
    print(data.shape[0], len(end_dates))
    data['end_timestamps'] = end_dates
    data['compare_closes'] = compare_closes
    data['pred'] = targets

    # remove dropped indices from the dropped_idx array
    data = data.loc[~data.index.isin(dropped_idx)]
    transformed_data = transformed_data.loc[~transformed_data.index.isin(dropped_idx)]
    # reset indices
    data.reset_index(inplace=True, drop=True)
    transformed_data.reset_index(inplace=True, drop=True)
    # count number of GAP target variables
    same_price_idx = data[data['pred'] == 'GAP'].index.values
    data = data.loc[~data.index.isin(same_price_idx)]
    transformed_data = transformed_data.loc[~transformed_data.index.isin(same_price_idx)]
    # insert original target variables into transformed dataframe
    transformed_data['pred'] = data['pred']
    transformed_data['end_timestamps'] = data['end_timestamps']
    transformed_data['compare_closes'] = data['compare_closes']
    return transformed_data

def _produce_prediction(data, window):
    """create predicted target values"""

    # custom mapping for trading window at given intraday time
    map: dict = {
        '09:30:00': 13, '10:00:00': 12, '10:30:00': 11, '11:00:00': 10, '11:30:00': 9, "12:00:00": 8,
        '12:30:00': 7, '13:00:00': 6, '13:30:00': 5, '14:00:00': 4, '14:30:00': 3, '15:00:00': 2,
        '15:30:00': 1, '16:00:00': 0
    }
    # create
    times, closes = data['time'], data['close']
    targets = np.array([])

    for start_idx in range(data.shape[0]):
        # get prices to compare for target variable
        current_close = closes.iloc[start_idx]
        try:
            compare_close = closes.iloc[start_idx + window + map[times.iloc[start_idx]]]
        except IndexError:
            compare_close = np.nan

        # check if larger or smaller
        if np.isnan(compare_close):
            targets = np.append(targets, np.nan)
        else:
            if current_close < compare_close:
                targets = np.append(targets, 1)
            else:
                targets = np.append(targets, 0)
    data['pred'] = targets
    return data['pred']

def test_boundaries_whole(df, ranges, training_data_boundaries):
    boundary_flag: bool = True
    for var, range in zip(['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k'], ranges):
        if training_data_boundaries(range[0], range[1], df[var]):
            boundary_flag = False
    return boundary_flag

def live_accuracy(trends, strikes, comp_price):
    """check the live trades accuracy"""

    # create output container
    container: list = []

    for i, (trend, strike) in enumerate(zip(trends, strikes)):
        if i <= 13:
            if trend == 1:
                if strike < comp_price:
                    container.append(True)
                else:
                    container.append(False)
            else:
                if strike > comp_price:
                    container.append(True)
                else:
                    container.append(False)
        else:
            container.append('-')
    return container

features = ['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k']
def draw_distributions(original, target, new_original_weights, plot=False):
    ks_vals: list = []
    for id, column in enumerate(features, 1):
        if plot:
            plt.figure(figsize=[15, 7])
            plt.subplot(2, 3, id)
            sns.kdeplot(data=original, x=column, color='red', weights=new_original_weights)
            sns.kdeplot(data=target, x=column, color='blue')
            plt.title(column)
            plt.show()
        ks_vals.append(ks_2samp_weighted(original[column], target[column],
                                         weights1=new_original_weights, weights2=np.ones(len(target), dtype=float)))
    return sum(ks_vals)/len(ks_vals)

ticker = yf.Ticker('SPY')
yesterday_price_backtest = ticker.history(interval='1d', start="2020-01-01", end="2022-01-21")['Close']
nyse = mcal.get_calendar('NYSE')

def evaluate_preds(strike, close, trend):
  """get actual price movements to live data"""
  if strike < close and trend == 1:
    return True
  elif strike > close and trend == 0:
    return True
  else:
    return False

from datetime import timedelta
from datetime import datetime
def get_n_market_days_ahead(n, start_date):
  """get the next market day in an n-day interval"""
  start_date = str(start_date)
  start_date = datetime.strptime(start_date, "%Y-%m-%d")
  end_date = start_date + timedelta(days=15)

  early = nyse.schedule(start_date=start_date, end_date=end_date)
  actual_end = str(early.iloc[:n+1].iloc[-1]['market_open']).split(' ')[0]
  return yesterday_price_backtest[yesterday_price_backtest.index == str(actual_end)].iloc[0]

def backtest_func(live_data, model, threshold):
  """backtesting function"""
  live_data['trend'] = model.predict(live_data[features])
  live_data['day'] = live_data['start_date'].apply(lambda x: str(x).split(' ')[0])
  live_data['close_price'] = live_data['day'].apply(lambda x: get_n_market_days_ahead(params['window']//14, str(x)))

  live_data['actual'] = live_data.apply(lambda x: evaluate_preds(x.strikes, x.close_price, x.trend), axis=1).astype(int)
  live_data.drop('day', axis=1, inplace=True)
  print(live_data[live_data['prob'] > threshold]['actual'].value_counts())

# backtesting mode turn on/off
backtest: bool = True

def create_model(years=params['years'], trend=params['window'], wavelet=params['wavelet'], facade=False):
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
    start = perf_counter()
    data['pred'] = _produce_prediction(copy_data, window=trend)
    print("Target Variable Creation Runtime", perf_counter() - start)

    # create live data samples from equivalent training distribution
    live_data = data[-trend:].drop('pred', axis=1)[['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k']]
    strike_prices = copy_data[-trend:]['close']
    live_dates = data[-trend:]['timestamp']

    # resolve formatting and nan issues
    del (data['close'])
    data.dropna(inplace=True)
    print(data.shape, 'tracking data shape 4')
    #data = data.sample(frac=1).reset_index(drop=True)
    features = ['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k']
    target = 'pred'

    # train-test data splits
    test_ratio = 0.30
    #X_train, X_test, y_train, y_test = train_test_split(data[features], data[target],
    #                                                    test_size=test_ratio, random_state=42)
    X_train, X_test = data[features][:int(0.8*data.shape[0])], data[features][int(0.8*data.shape[0]):]
    y_train, y_test = data[target][:int(0.8*data.shape[0])], data[target][int(0.8*data.shape[0]):]

    # transform distributions into the normal distribution to standardize training-testing-live samples
    live_data = live_data[features]
    transform: bool = False
    if transform:
        output_dist = 'uniform'
        train_trans = quantile_transform(X_train, output_distribution=output_dist, n_quantiles=len(X_train))
        test_trans = quantile_transform(X_test, output_distribution=output_dist, n_quantiles=len(X_train))
        live_trans = quantile_transform(live_data, output_distribution=output_dist, n_quantiles=len(X_train))
        for i, col in enumerate(X_train.columns):
            X_train[col] = train_trans[:, i]
            X_test[col] = test_trans[:, i]
            live_data[col] = live_trans[:, i]

    reweight_check = False
    opt_weights = np.ones(len(X_train))
    if reweight_check and not transform:
        opt_weights, distance = None, 10000
        for estimator in tqdm([150, 200, 250]):
            for lr in [0.05, 0.1]:
                for depth in [3, 4]:
                    for leaf in [1000, 2000, 3000]:
                        reweighter = reweight.GBReweighter(n_estimators=estimator, learning_rate=lr, max_depth=depth,
                                                           min_samples_leaf=leaf)
                        reweighter.fit(X_train, live_data[features])
                        gb_weights_test = reweighter.predict_weights(X_train)
                        dist_distance = draw_distributions(X_train, live_data, gb_weights_test)
                        if dist_distance < distance:
                            print(dist_distance, distance)
                            opt_weights = gb_weights_test
                            distance = dist_distance

    print(draw_distributions(X_train, live_data, opt_weights, plot=False)) ### DISABLE PLOTTING WHEN TUNING
    ratio = Counter(y_train)
    scale_pos_weight = max(ratio.values()) / min(ratio.values())
    model = XGBClassifier(scale_pos_weight=scale_pos_weight).fit(X_train, y_train, sample_weight=opt_weights)
    #model = RandomForestClassifier(class_weight='balanced').fit(X_train, y_train)

    stat, p_value = ttest_ind(X_train, live_data)
    print("T-test P-value: ", p_value)

    # determine accuracy and append to results
    rf_prediction = model.predict(X_test)
    print("Raw Accuracy: ", accuracy_score(list(y_test), list(rf_prediction)))
    print("ROC AUC: ", roc_auc_score(y_test, rf_prediction))
    print('F1 Score: ', f1_score(y_test, rf_prediction, average='macro'))
    rf_accuracy = balanced_accuracy_score(list(y_test), list(rf_prediction))
    print('Test Accuracy: ', rf_accuracy)
    print(years, trend, wavelet)

    # make live predictions
    live_predictions = model.predict(live_data)
    live_prob_predictions = model.predict_proba(live_data)
    live_data['pred'] = live_predictions
    live_data['strikes'] = strike_prices
    live_data['prob'] = [i[np.argmax(i)] for i in live_prob_predictions]
    live_data['date'] = live_dates
    live_data['window'] = [trend // 14 for _ in range(live_data.shape[0])]
    live_data['day'] = live_data['date'].apply(lambda x: str(x).split(' ')[0])
    live_data = live_data[live_data['day'].isin(live_data['day'].unique()[-3:])].drop('day', axis=1)
    live_data.rename(columns={'pred': 'trend', 'date': 'start_date'}, inplace=True)

    # measure accuracy across different confidence filters
    correct, incorrect, skipped = 0, 0, 0
    threshold = 0.9
    try:
        for pred, prob, actual in zip(rf_prediction, model.predict_proba(X_test), y_test):
            if prob[int(pred)] >= threshold:
                if pred == actual:
                    correct += 1
                else:
                    incorrect += 1
            else:
                skipped += 1
        print((f"Acc: {correct / (correct + incorrect)}"), f'Skip Ratio: {skipped / len(y_test)}')
    except ZeroDivisionError:
        pass

    # create a backtest
    if backtest:
        backtest_func(live_data, model, 0.9)

    if not facade:
        live_data.to_csv('live_data_backtest.csv')
        print(live_data[['window', 'trend', 'start_date', 'prob', 'strikes']][live_data['prob'] > 0.9])

    # report the NNI console
    nni.report_final_result(rf_accuracy)
    #nni.report_intermediate_result(cross_val_acc)

    if not facade:
        return model, rf_accuracy, live_data[['trend', 'prob', 'strikes', 'start_date']], live_data[features], \
               X_train, X_test, y_train, y_test, data
    else:
        return model, accuracy_score(list(y_test), list(rf_prediction)), \
               live_data[['window', 'trend', 'start_date', 'prob', 'strikes']][live_data['prob'] > 0.9]

create_model()
#create_model(years=3, trend=42, wavelet='coif2')