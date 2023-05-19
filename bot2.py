from binance import Client
import time
from bot_functions import *

APIKEY = ""
SECRETKEY = ""

client = Client(APIKEY, SECRETKEY)
print("logged in")



def bot2(model):
    print("Bot working")
    open_pos = False
    initial_cap = get_asset_balance(client, 'TUSD')
    counter = 0
    trades = 0
    while True:
        symbol1 = "BTCTUSD"
        data = get_1m_data(symbol1, client.KLINE_INTERVAL_5MINUTE, lookback='300 min ago')
        data = apply_ta(data)
        data = get_model_data(data)
        
        tusd_balance = get_asset_balance(client, 'TUSD')
        btc_balance = get_asset_balance(client, 'BTC')
        predictors = ["open", "close", "low", "high","gap" , "macd" 
                  ,"14_MA", "Boll", "3MA_diff", "7MA_diff", "14MA_diff", 
                "21MA_diff",  "RSI", "streak", "neg_streak"]
        current_price_btc = get_current_price(client, symbol1)
        gains = tusd_balance - initial_cap
        counter += 1
        if counter%10 == 0:
            print (f"Total gains: {gains}")
            print(tusd_balance) 
        

        #BUYING
        if not open_pos and are_min_above_5_10():
            print(data.iloc[-1])
            print("Position to be opened...")
            if make_trade(data.iloc[-1], model, predictors):
                
                order = client.create_order(symbol = symbol1,
                                                side = 'BUY',
                                                type = 'MARKET',
                                                quantity = truncate(tusd_balance/get_current_price(client, symbol1)))
                buy_price = float(order['fills'][0]['price'])
                trade_time = time.time()
                print(f"Buy Price: {buy_price}")
                open_pos = True

            
            else:
                print("Position not opened")

        #SELLING

        else:
            time.sleep(2)
            if open_pos:
                #sell btc
                if current_price_btc < buy_price*0.99999 or current_price_btc > buy_price*1.001:
                    order = client.create_order(symbol = symbol1,
                                                    side = 'SELL',
                                                    type = 'MARKET',
                                                    quantity = truncate(btc_balance))
                    sell_price = float(order['fills'][0]['price'])
                    print(f"Sell Price: {sell_price}")
                    open_pos = False
                    time.sleep(5)
                    trades += 1
                    print(f"Trades: {trades}")
                    
                elif minute_passed_since_trade_5m(trade_time):
                    order = client.create_order(symbol = symbol1,
                                                    side = 'SELL',
                                                    type = 'MARKET',
                                                    quantity = truncate(btc_balance))
                    sell_price = float(order['fills'][0]['price'])
                    print(f"Sell Price: {sell_price}")
                    open_pos = False 
                    time.sleep(5)
                    trades += 1
                    print(f"Trades: {trades}")

print("Starting BOT...")
model = apply_all_bot2()
bot2(model)
    