import pandas as pd
import numpy as np
from binance import Client
import ta
from ta.volatility import BollingerBands
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import time
import datetime

APIKEY = ""
SECRETKEY = ""

client = Client(APIKEY, SECRETKEY)

def get_1m_data(symbol, time, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, 
                                                    time,
                                                    lookback))

    frame = frame.iloc[:,:6]
    frame.columns = ['Time', 'open', 'high','low','close', 'volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit= 'ms')
    frame = frame.astype(float)
    return frame

def apply_ta(df):

    df['RSI']= ta.momentum.rsi(df.close, window=14)
    df['macd']= ta.trend.macd_diff(df.close)
    df['21_MA'] = ta.trend.sma_indicator(df['close'], window=21)
    df['14_MA'] = ta.trend.sma_indicator(df['close'], window=14)
    df['7_MA'] = ta.trend.sma_indicator(df['close'], window=7)
    df['3_MA'] = ta.trend.sma_indicator(df['close'], window=3)
    # df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=100).average_true_range()
    indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df['Boll']= indicator_bb.bollinger_wband()
    df.dropna(inplace=True)
    return df

def calculate_streak(df):
    streak = np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0
    return streak


def calculate_neg_streak(df):
    streak = np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        if df['close'].iloc[i] < df['close'].iloc[i-1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0
    return streak

def get_model_data(df):   
    df["gap"] = ((df["close"] - df["open"])/df["close"])
    df["next"] = df["close"].shift(-1)
    df["target"] = (df["next"] > df["close"]).astype(int)
    df["3MA_diff"] = (df["3_MA"] - df["close"])/df["close"] 
    df["7MA_diff"] = (df["7_MA"] - df["close"])/df["close"]   
    df["14MA_diff"] = (df["14_MA"]- df["close"])/df["close"] 
    df["21MA_diff"] = (df["21_MA"] - df["close"])/df["close"] 
    df["streak"] = calculate_streak(df)
    df["neg_streak"] = calculate_neg_streak(df)
    return df


def get_model(df):
    model1 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=50, random_state=1)
    model2 = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_split=40, random_state=2)
    voting_clf = VotingClassifier(estimators=[('rf1', model1), ('rf2', model2)],  voting='hard')
    predictors = ["open", "close", "low", "high","gap" , "macd" 
                  ,"14_MA", "Boll", "3MA_diff", "7MA_diff", "14MA_diff", 
                "21MA_diff",  "RSI", "streak", "neg_streak"]
    voting_clf.fit(df[predictors], df["target"])

    return voting_clf

def get_asset_balance(client, asset):
    asset_balance = client.get_asset_balance(asset=asset)
    return float(asset_balance['free'])

def make_trade(data, model, predictors):
    # Create a DataFrame with the correct feature names
    data_df = pd.DataFrame([data[predictors].values], columns=predictors)
    prediction = model.predict(data_df)
    return True if prediction[0] == 1 else False

def minute_passed_since_trade(trade_time):
    current_time = time.time()
    elapsed_time = current_time - trade_time
    return elapsed_time >= 60

def minute_passed_since_trade_5m(trade_time):
    current_time = time.time()
    elapsed_time = current_time - trade_time
    return elapsed_time >= 60*4.9

def get_current_price(client, symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

#max precision 0.001
def truncate(number):
    return int(number * 1000) / 1000

def are_seconds_above_50():
    now = time.localtime()
    return now.tm_sec > 54


def are_min_above_5_10():
    now = time.localtime()
    if (now.tm_min%10 == 4) or (now.tm_min%10 == 9):
        if now.tm_sec > 53:
         return True
    else:
        return False

def apply_all():
    print("Getting data...")
    symbol1 = "BTCUSDT"
    btc = get_1m_data(symbol1, client.KLINE_INTERVAL_1MINUTE, lookback='8000 min ago')
    btc = apply_ta(btc)
    btc = get_model_data(btc)
    print("Making model...")
    model = get_model(btc)
    return model

def apply_all_bot2():
    print("Getting data...")
    symbol1 = "BTCUSDT"
    btc = get_1m_data(symbol1, client.KLINE_INTERVAL_5MINUTE, lookback='40000 min ago')
    btc = apply_ta(btc)
    btc = get_model_data(btc)
    print("Making model...")
    model = get_model(btc)
    print("Model Ready")
    return model