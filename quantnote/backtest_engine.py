import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import sqlite3
import time
import multiprocessing
from itertools import product
import matplotlib.pyplot as plt

def timeis(func):  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('[{}] is executed in {:.2f} seconds'.format(func.__name__, end-start))
        return result
    return wrap

def query_df(df, ticker):
    return df[df.ticker.values == ticker]
           
def divide_by_ticker(df, pool):
    if pool is not None:
        keys = list(set(df.ticker))
        values = pool.starmap(query_df, product([df],keys))
        return dict(zip(keys, values))
    else:
        return {ticker:df[df.ticker.values == ticker] for ticker in set(df.ticker)}
    

class BacktestEngine():
    def __init__(self, db_name=None, num_process=4):
        self.db = db_name
        self.cache = {}
        self.initialize(num_process)
        
    @timeis
    def initialize(self, num_process):
        print('Loading DB...')
        stime = time.time()

        db = sqlite3.connect(self.db)

        try:
            msg = 'SELECT * FROM universe'
            df = pd.read_sql(msg,db).set_index('date')
            df.index = pd.to_datetime(df.index)
            universe_df = df.copy().sort_index()

            msg = 'SELECT * FROM ticker'
            df = pd.read_sql(msg,db).set_index('permaticker')
            ticker_df = df.copy().sort_index()

            msg = 'SELECT * FROM fundamentals'
            df = pd.read_sql(msg,db).set_index('datekey')
            df.index = pd.to_datetime(df.index)
            fundamental_df = df.copy().sort_index()

            msg = 'SELECT * FROM metric'
            df = pd.read_sql(msg,db).set_index('date')
            df.index = pd.to_datetime(df.index)
            metric_df = df.copy().sort_index()

        except:
            self.cache['universe'] = {}
            self.cache['ticker'] = {}
            self.cache['fundamentals'] = {}
            self.cache['metric'] = {}
            print('Quandl data does not exist')

        try:
            msg = 'SELECT * FROM macro'
            df = pd.read_sql(msg,db).set_index('datekey')
            df.index = pd.to_datetime(df.index)
            df['value'] = pd.to_numeric(df['value'])
            df['cdate'] = pd.to_datetime(df['cdate'])
            macro_df = df.copy().sort_index()
        except:
            self.cache['macro'] = {}
            print('FRED data does not exsit')

        
        msg = 'SELECT * FROM market'
        df = pd.read_sql(msg,db).set_index('date')
        df.index = pd.to_datetime(df.index)
        market_df = df.copy().sort_index()

        db.close
        etime = time.time()
        print('DB loaded in {:.2f} seconds'.format(etime-stime))
        
        pool = multiprocessing.Pool(num_process)

        try:
            self.cache['macro'] = divide_by_ticker(macro_df, None)
        except:
            pass
        try:
            self.cache['universe'] = universe_df
            self.cache['tickerinfo'] = divide_by_ticker(ticker_df, None)
            self.cache['fundamentals'] = divide_by_ticker(fundamental_df, None)
            self.cache['metric'] = divide_by_ticker(metric_df, pool)
            self.cache['market'] = divide_by_ticker(market_df, pool)
        except:
            pass
        pool.close()
        
    def get_universe(self, date):
        if self.custom_universe is not None:
            return self.custom_universe
        else:
            try:
                last_date = self.cache['universe'].loc[:date].index[-1]
                universe = list(set(self.cache['universe'].loc[last_date].ticker))
            except:
                assert False
            return universe

    @timeis
    def run_backtest(self, strategy, sdate, edate, period='M',
                    transaction_cost=0, verbose=False, custom_universe=None):
        start_T = time.time()
        ETA_past = 0
        self.custom_universe = custom_universe
        self.asset = {}
        self.qty = {}
        self.avgprice = {}
        self.transaction = {}
        
        sdate = mcal.get_calendar('NYSE').valid_days(
            start_date='1900-01-01', end_date=sdate)[-1].strftime('%Y-%m-%d')
        edate = mcal.get_calendar('NYSE').valid_days(
            start_date='1900-01-01', end_date=edate)[-1].strftime('%Y-%m-%d')

        self.bdates = mcal.get_calendar('NYSE').valid_days(
            start_date=sdate, end_date=edate)
        self.bdates = [x.tz_localize(None) for x in self.bdates]

        print('Backtest period: {} -- {}'.format(self.bdates[1], self.bdates[-1]))

        date = self.bdates[0]
        self.asset[date] = {'cash':1}
        self.avgprice[date] = {'cash':1}
        self.qty[date] = {'cash':1}
        universe_list = self.get_universe(date)
        self.delisted_tickers = []

        target_weight = self.compute_target(date, universe_list, strategy)
        is_rebal = True

        for date in self.bdates[1:]:
            self.update_asset(date, verbose)
            self.liquidate_delisted_tickers(date)
            
            if is_rebal:
                self.rebalance_asset(date, target_weight, transaction_cost)
                
            ETA = time.time()-start_T
            if ETA - ETA_past > 1:
                print('===','date:{}'.format(date),'/',
                      'total_asset:{:.3f}'.format(sum(self.asset[date].values())),'/',
                      'time elapsed:{:.1f}'.format(ETA),'===',
                      end='\r')
                ETA_past = ETA

            is_rebal = self.set_rebal_condition(date, period)
               
            if is_rebal:
                universe_list = self.get_universe(date)
                universe_list = list(set(universe_list)-set(self.delisted_tickers))
                self.delisted_tickers = []
                target_weight = self.compute_target(date, universe_list, strategy)

        ETA = time.time()-start_T   
        print('===','date:{}'.format(date),'/',
                      'total_asset:{:.3f}'.format(sum(self.asset[date].values())),'/',
                      'time elapsed:{:.1f}'.format(ETA),'===')
                      
        self.asset_df = pd.DataFrame(self.asset).T.fillna(0).iloc[1:]
        self.transaction_df = pd.DataFrame(self.transaction).T.fillna(0)
        self.avgprice_df = pd.DataFrame(self.avgprice).T.fillna(0)
        self.qty_df = pd.DataFrame(self.qty).T.fillna(0)

    def sanity_check(self, date):
        asset_keys = self.asset[date].keys()
        qty_keys = self.qty[date].keys()
        avgprice_keys = self.avgprice[date].keys()

        assert set(asset_keys) == set(qty_keys)
        assert set(asset_keys) == set(avgprice_keys)

    def update_asset(self, date, verbose):
        yesterday = self.bdates[self.bdates.index(date)-1]
        self.asset[date] = {
            ticker : self.asset[yesterday][ticker]*self.get_return(ticker, date, verbose)
            for ticker in self.asset[yesterday]}
        self.avgprice[date] = self.avgprice[yesterday]
        self.qty[date] = self.qty[yesterday]
        self.transaction[date] = {}
        self.sanity_check(date)

    def rebalance_asset(self, date, target_weight, transaction_cost):
        current_asset = self.asset[date].copy()
        current_avgprice = self.avgprice[date].copy()
        current_qty = self.qty[date].copy()

        revised_weight = target_weight.copy()
        for ticker in target_weight.keys():
            try:
                ticker_price = self.get_price(ticker, date)
            except:
                revised_weight.pop(ticker)
        wgt_sum = sum(revised_weight.values())

        total_asset = sum(current_asset.values())
        target_asset = {ticker:total_asset*revised_weight[ticker]/wgt_sum for ticker in revised_weight}
        transaction_asset = {}
        updated_asset = {}
        updated_avgprice = {}
        updated_qty = {}

        for ticker in set(target_asset.keys()).union(set(current_asset.keys())):
            ticker_price = self.get_price(ticker, date)
            qty = current_qty[ticker] if ticker in current_qty else 0
            avgprice = current_avgprice[ticker] if ticker in current_avgprice else 0

            target = target_asset[ticker] if ticker in target_asset else 0
            current = current_asset[ticker] if ticker in current_asset else 0
            transaction = target - current

            if transaction > 0 :
                transaction = (1-transaction_cost)*transaction
                buy_qty = transaction / ticker_price
                updated_qty[ticker] = qty + buy_qty
                updated_avgprice[ticker] = (qty*avgprice + buy_qty*ticker_price) / (qty + buy_qty)
                updated_asset[ticker] = (qty + buy_qty) * ticker_price

            elif transaction <= 0:
                sell_qty = transaction / ticker_price
                if ticker in target_asset.keys():
                    updated_qty[ticker] = qty + sell_qty
                    updated_avgprice[ticker] = avgprice
                    updated_asset[ticker] = (qty + sell_qty) * ticker_price

            transaction_asset[ticker] = transaction
            
        assert np.abs(1- sum(target_asset.values())/total_asset) < 1e-6

        self.asset[date] = updated_asset
        self.avgprice[date] = updated_avgprice
        self.qty[date] = updated_qty
        self.transaction[date] = transaction_asset

        self.sanity_check(date)

    def get_price(self, ticker, date):
        try:
            if ticker == 'cash':
                ticker_price = 1 
            elif ticker in self.cache['market'].keys():
                ticker_price = self.cache['market'][ticker].close.loc[date]
        except:
            assert False
        return ticker_price

    def get_return(self, ticker, date, verbose):
        if ticker == 'cash':
            return 1
        try:
            curr_price = self.cache['market'][ticker]['close'].loc[date]
            last_price = self.cache['market'][ticker]['close'].shift().loc[date]
            return curr_price/last_price
        except:
            if verbose: 
                print('\n')
                print('{} is delisted at {}'.format(ticker, date) + '\n')
            self.delisted_tickers.append(ticker)
            return 1

    def liquidate_delisted_tickers(self, date):
        if 'cash' not in self.asset[date]: 
            self.asset[date]['cash'] = 0
            self.avgprice[date]['cash'] = 1
            self.qty[date]['cash'] = 0

        for ticker in self.delisted_tickers:
            if ticker in self.asset[date]:
                self.asset[date]['cash'] += self.asset[date][ticker]
                self.qty[date]['cash'] += self.asset[date][ticker]

        for ticker in self.delisted_tickers:
            self.asset[date].pop(ticker, None)
            self.avgprice[date].pop(ticker, None)
            self.qty[date].pop(ticker, None)

    def set_rebal_condition(self, date, period):
        try:
            tomorrow = self.bdates[self.bdates.index(date)+1]
        except:
            tomorrow = date

        if period == 'D':
            is_rebal = True 
        elif period == 'W':
            is_rebal = (date.weekday() == 4)
        elif period == 'M':
            is_rebal = (tomorrow.month != date.month)
        else:
            is_rebal = False

        return is_rebal

    def compute_target(self, date, universe_list, strategy):
        date = date.strftime('%Y-%m-%d')
        strategy.date = date
        cache = {}
        for table in self.cache:
            cache[table] = {}
            for ticker in self.cache[table]:
                cache[table][ticker] = self.cache[table][ticker].loc[:date]
        strategy.cache = cache
        target_weight = strategy.compute_target(universe_list)

        return target_weight

    def show_report(self, benchmark='^SP500TR', filename=None):
        
        asset = self.asset_df.sum(axis=1)
        if benchmark is not None and type(benchmark) == str:
            dates = list(set(self.asset_df.index) & set(self.cache['market'][benchmark].index))
            dates.sort()
            bench = self.cache['market'][benchmark].close.loc[dates]
            bench = bench/bench.iloc[0]

        elif benchmark is not None and type(benchmark) == type(pd.DataFrame()):
            dates = list(set(self.asset_df.index) & set(benchmark.index))
            dates.sort()
            benchmark = benchmark.sum(axis=1)
            bench = benchmark.loc[dates]
            bench = bench/bench.iloc[0]

        stat_df = self.stat(bench)

        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]},figsize=(12,5))
        
        ax1 = axs[0]
        if benchmark is not None:
            ax1.plot(asset)
            ax1.plot(bench)
            ax1.legend(['Strategy', 'Benchmark'])
        else:
            ax1.plot(asset)
            ax1.legend(['Strategy'])
        
        ax2 = axs[1]
        ax2.axis('off')
        table = ax2.table(cellText=stat_df.values, rowLabels = stat_df.index, 
                            colLabels=stat_df.columns, bbox=[0,0,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        fig.autofmt_xdate()
        fig.show()

        if filename is not None:
            fig.savefig(filename)

    def stat(self, bench):
        def _max_underwater(asset):
            ser = (asset - asset.cummax())
            underwater_periods = []
            days = 0
            for item in ser:
                if item < 0:
                    days += 1
                else:
                    underwater_periods = underwater_periods+[days]
                    days=0
            return np.max(underwater_periods)
            
        asset = self.asset_df.sum(axis=1).iloc[1:]
        transaction = self.transaction_df.apply(np.abs).sum(axis=1).iloc[1:]

        TO = (transaction/asset).resample('M').sum().mean()*6

        if bench is not None:
            dates = list(set(asset.index) & set(bench.index))
            dates.sort()
            asset = asset.loc[dates]
            transaction = transaction.loc[dates]
            bench = bench.loc[dates]
        
        rets = asset.apply(np.log).diff().dropna()
        cagr = 252*np.mean(rets)
        vol = np.sqrt(252)*np.std(rets)
        sharpe = np.sqrt(252)*np.mean(rets)/np.std(rets)
        IR = 0
        mdd = asset.div(asset.cummax()).min()-1
        mup = _max_underwater(asset)
        

        if bench is not None:

            bench_rets = bench.apply(np.log).diff().dropna()
            excess_rets = rets - bench_rets
            IR = np.sqrt(252)*np.mean(excess_rets)/np.std(excess_rets)

            cagr_b = 252*np.mean(bench_rets)
            vol_b = np.sqrt(252)*np.std(bench_rets)
            sharpe_b = np.sqrt(252)*np.mean(bench_rets)/np.std(bench_rets)
            IR_b = 0
            mdd_b = bench.div(bench.cummax()).min()-1
            mup_b = _max_underwater(bench)

            stat_b = {'CAGR':'{:.3f}'.format(cagr_b), 
                'Vol':'{:.3f}'.format(vol_b), 
                'Sharpe':'{:.3f}'.format(sharpe_b), 
                'IR':'{:.3f}'.format(IR_b), 
                'MDD':'{:.3f}'.format(mdd_b), 
                'MUP':'{} days'.format(mup_b),
                'Turnover':'{:.3f}'.format(0)}
        
        stat = {'CAGR':'{:.3f}'.format(cagr), 
                'Vol':'{:.3f}'.format(vol), 
                'Sharpe':'{:.3f}'.format(sharpe), 
                'IR':'{:.3f}'.format(IR), 
                'MDD':'{:.3f}'.format(mdd), 
                'MUP':'{} days'.format(mup),
                'Turnover':'{:.3f}'.format(TO)}

        stat_df = pd.Series(stat).to_frame()
        stat_df.columns = ['Strategy']

        if bench is not None:
            stat_df_b = pd.Series(stat_b).to_frame()
            stat_df = pd.concat([stat_df, stat_df_b], axis=1)
            stat_df.columns = ['Strategy', 'Benchmark']

        return stat_df

    def show_sample_strategy(self):
        tab = "    "
        line = "class myStrategy(Strategy):\n"
        line += tab+"def __init__(self):\n"
        line += tab+tab+"super().__init__()\n"
        line += "\n"
        line += tab+"def compute_target(self, universe_list):\n"
        line += tab+tab+"target_weight = { }\n"
        line += tab+tab+"for ticker in universe_list:\n"
        line += tab+tab+tab+"target_weight[ticker] = 1\n"
        line += tab+tab+"target_weight = self.normalize(target_weight)\n"
        line += tab+tab+"return target_weight\n"
        line += "\n"
        line += tab+"def custom_factor(self, ticker, ftype):\n"
        line += tab+tab+"if ftype == 'random':\n"
        line += tab+tab+tab+"return np.random.rand()\n"
        line += tab+tab+"else:\n"
        line += tab+tab+tab+"assert False\n"
        
        print(line)