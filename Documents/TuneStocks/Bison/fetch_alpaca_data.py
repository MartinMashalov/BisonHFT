"""fetch alpaca data from api"""
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from requests import get
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# create starting timestamps
years = 25
end = (date.today()-timedelta(days=0)).strftime("%Y-%m-%d")
date_1 = datetime.strptime(end, "%Y-%m-%d")
start_1 = date_1 - timedelta(days=365*years)
start = str(start_1).replace(' 00:00:00', '')
main_df = pd.DataFrame(data=[], columns=['Timestamp', 'Open', "High", 'Low', 'Close', 'Volume'])

# check if download was valid
def check_download(results: dict):
    """check if minutes are off during download stage"""
    first_timestamp = results['results'][0]['t']
    minute = datetime.utcfromtimestamp(first_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S').split(':')[1]
    return (minute == '30' or minute == "00")

def polygon_to_df(results: dict):
    """download json results to dataframe"""
    container = []
    for result in results['results']:
        container.append([datetime.utcfromtimestamp(result['t']/1000).strftime('%Y-%m-%d %H:%M:%S'),
                      result['o'], result['h'], result['l'], result['c'], result['v']])
    df = pd.DataFrame(data=container, columns=['Timestamp', 'Open', "High", 'Low', 'Close', 'Volume'])
    return df

# TO DO: another while loop to collect all data up to the end-date
def download_polygon(start=start, start_1=start_1, main_df=main_df, interval: str = 'minute', interval_length: int = 30):
    counter = 0
    while start < end and counter < 100:
        download_correct: bool = False
        print('Start Date: ', start)
        while not download_correct:
            url = f'https://api.polygon.io/v2/aggs/ticker/SPY/range/{interval_length}/{interval}/{start}/{end}?adjusted=false&sort=asc&limit=50000&apiKey=eTD2a0gOvakkPjBpyYBqRgiWY9CLJ0ot'
            results: dict = get(url).json()
            download_correct = check_download(results)
            if not download_correct:
                start_1 = start_1 + timedelta(days=1)
                start = str(start_1).replace(' 00:00:00', '')
        main_df = main_df.append(polygon_to_df(results))
        start = main_df.iloc[-1]['Timestamp'].split(' ')[0]
        counter += 1
    return main_df

# filter out before and after market times to preserve only intraday entries
def get_hour_min(hour_str: str):
    """get hour and minute from hour"""
    hour, min, sec = hour_str.split(':')
    return int(hour), int(min)

# subdivide the downloaded dataframe into years
def subdivide_df(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """subdivide the pandas dataframe by years into the past"""

    # create new dataframe features for time and year
    df['Year'] = df['Timestamp'].apply(lambda x: x.split(' ')[0])
    df['time'] = df['Timestamp'].apply(lambda x: x.split(' ')[1])
    #df.drop('Date', axis=1, inplace=True)

    # make year column a dataframe index
    df.set_index('Year', inplace=True)

    # calculate time y-years from present
    today = date.today()
    end = today.strftime("%Y-%m-%d")
    dtObj = datetime.strptime(str(end), "%Y-%m-%d")
    past_date = dtObj - relativedelta(years=years)

    # filter dataframe for results
    df = df[str(past_date).split(' ')[0]:end]
    df['day'] = df.index
    df.reset_index(inplace=True)
    df.drop('Year', 1, inplace=True)
    return df

def fetch_alpaca(years: int, stacked_df) -> None:
    """fetch alpaca data from backend client and stacking algorithm"""
    df = subdivide_df(stacked_df, years)
    df['hour'] = df['time'].apply(lambda x: get_hour_min(x)[0])
    df['minute'] = df['time'].apply(lambda x: get_hour_min(x)[1])

    # remove after-market and before-market data
    sub_df = df[(df['hour'] >= 9) & (df['hour'] <= 16)]

    # get datasets to remove before 9:30:00 and over 16:00:00
    sub9_df = sub_df[(sub_df['hour'] == 9) & (sub_df['minute'] < 30)]
    sub16_df = sub_df[(sub_df['hour'] == 16) & (sub_df['minute'] > 0)]

    # drop them from the big dataset
    sub_df.drop(sub9_df.index, axis=0, inplace=True)
    sub_df.drop(sub16_df.index, axis=0, inplace=True)
    df = sub_df
    df.reset_index(inplace=True)

    # drop unused columns
    df.drop(['hour', 'minute'], axis=1, inplace=True)

    # remove missing days from dataframe
    missing_days = []
    value_counts = df['day'].value_counts()
    for key, value in value_counts.items():
        if value != 14:
            missing_days.append(key)
    copy_data = df.loc[~df['day'].isin(missing_days)]
    copy_data.drop('index', 1, inplace=True)

    # download as csv file in directory
    copy_data.to_csv(f'data_yearly/{years}_years_alpaca.csv')

def fetch_alpaca_csv(years: int) -> pd.DataFrame:
    """fetch alpaca data for model training purposes"""
    return pd.read_csv(f'data_yearly/{years}_years_alpaca.csv')

def filter_daily_data(df) -> pd.DataFrame:
    """filter out and clean daily data"""

    # create minute and hourly intervals
    df['hour'] = df['time'].apply(lambda x: get_hour_min(x)[0])
    df['minute'] = df['time'].apply(lambda x: get_hour_min(x)[1])

    # remove after-market and before-market data
    sub_df = df[(df['hour'] >= 9) & (df['hour'] <= 16)]

    # get datasets to remove before 9:30:00 and over 16:00:00
    sub9_df = sub_df[(sub_df['hour'] == 9) & (sub_df['minute'] < 30)]
    sub16_df = sub_df[(sub_df['hour'] == 16) & (sub_df['minute'] > 0)]

    # drop them from the big dataset
    sub_df.drop(sub9_df.index, axis=0, inplace=True)
    sub_df.drop(sub16_df.index, axis=0, inplace=True)
    df = sub_df
    df.reset_index(inplace=True)

    # drop unused columns
    df.drop(['hour', 'minute'], axis=1, inplace=True)

    # filter out missing days

    return df

# download data
def fetch_action(intraday=True):
    # load in base dataframe
    if intraday:
        print('Intraday Data: ')
        stacked_df = download_polygon()
        for i in tqdm([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]):
            fetch_alpaca(i, stacked_df)
    else:
        stacked_df = download_polygon(interval='hour', interval_length=2)
        stacked_df.to_csv(f'data_daily/daily_alpaca.csv')

# fetch_action()