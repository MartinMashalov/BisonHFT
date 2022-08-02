import pandas as pd
import nni
import pandas_ta as pd_ta
from fetch_alpaca_data import fetch_alpaca_csv
from pandas import DataFrame
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score
import warnings
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
import yagmail

warnings.filterwarnings('ignore')

# set random seed on numpy
np.random.RandomState(42)
np.random.seed(42)

# performance csv headers
header = ['test_acc', 'live_acc']

params: dict = {
    'years': 3,
    'wavelet': 'sym6',
    'window': 42
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

# training data boundaries test to avoid extrapolation issues
training_data_boundaries = lambda min, max, val: True if val < min or val > max else False

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
    data = data.sample(frac=1).reset_index(drop=True)
    features = ['rsi', 'macd', 'willr', 'obv', 'proc', 'stoch_k']
    target = 'pred'

    # train-test data splits
    test_ratio = 0.20
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target],
                                                        test_size=test_ratio, random_state=42)

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

    # create and train xgboost decision tree model
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, seed=42).fit(X_train, y_train)

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

    # measure accuracy across different confidence filters
    correct, incorrect, skipped = 0, 0, 0
    threshold = 0.95
    for pred, prob, actual in zip(rf_prediction, model.predict_proba(X_test), y_test):
        if prob[pred] >= threshold:
            if pred == actual:
                correct += 1
            else:
                incorrect += 1
        else:
            skipped += 1
    print((f"Acc: {correct / (correct + incorrect)}"), f'Skip Ratio: {skipped / len(y_test)}')

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

    if not facade:
        print(live_data[['window', 'trend', 'start_date', 'prob', 'strikes']][live_data['prob'] > 0.9])

    # report the NNI console
    nni.report_final_result(rf_accuracy)
    #nni.report_intermediate_result(cross_val_acc)

    if not facade:
        return model, rf_accuracy, live_data[['trend', 'prob', 'strikes', 'start_date']], live_data[
            features], X_train, X_test, y_train, y_test, data
    else:
        return model, accuracy_score(list(y_test), list(rf_prediction)), \
               live_data[['trend', 'prob', 'strikes', 'start_date']]

create_model()