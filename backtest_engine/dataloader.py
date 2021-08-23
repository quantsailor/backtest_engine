import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import datetime as dt
from dateutil.relativedelta import relativedelta
import os, zipfile,sqlite3, time, requests, json

import yfinance as yf
import quandl 

def timeis(func):  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('[{}] is executed in {:.2f} seconds'.format(func.__name__, end-start))
        return result
    return wrap


DOWNLOAD_PATH = './data'
DB_PATH = './DB'
try:
    os.mkdir(DOWNLOAD_PATH)
except:
    pass
try:
    os.mkdir(DB_PATH)
except:
    pass

class DataLoader():
    def __init__(self, fred_key=None, quandl_key=None, fred_list=[], yfinance_list=[],
        sdate=None, edate=None, size=100, db_name=None, is_update=False):

        if sdate is None: sdate = '1900-01-01'
        if edate is None: edate = dt.datetime.today().strftime('%Y-%m-%d')
        
        if is_update:
            db = sqlite3.connect(DB_PATH+'/'+db_name)
            date_list = list(pd.read_sql('SELECT * FROM market', db).date.sort_values().unique())
            sdate = date_list[-1]
            db.close()

        print('Downloading data in {}--{} with marketcap screening={}'.format(
            sdate.replace('-','/'), edate.replace('-','/'), size))
        
        self.update_data(sdate, edate, size, fred_key, quandl_key, fred_list, yfinance_list)

        if db_name is None:
            db_name = DB_PATH+'/backtest_{}_{}_{}.db'.format(
                size, 
                pd.to_datetime(self.market_data_df.date.iloc[0]).strftime('%Y-%m-%d'), 
                pd.to_datetime(self.market_data_df.date.iloc[-1]).strftime('%Y-%m-%d'))
        else:
            db_name = DB_PATH+'/'+db_name

        self.save_db(db_name, is_update, is_local)

    @timeis
    def update_data(self, sdate, edate, size, fred_key, quandl_key, fred_list, yfinance_listt):
        if quandl_key is not None:
            quandl.ApiConfig.api_key = quandl_key
            self.universe_df = self.update_universe(sdate, edate, size)
            universe = list(set(self.universe_df.ticker))
            self.ticker_df = self.update_ticker(universe)
            self.quanterly_report_df = self.update_fundamentals(sdate, edate, universe)
            self.daily_metric_df = self.update_metric(sdate, edate, universe)
            self.market_data_df = self.update_market(sdate, edate, universe)

        if fred_key is not None:
            self.macro_df = self.update_macro(fred_key, fred_list)

        self.index_df = self.update_index(yfinance_listt)

    @timeis
    def update_universe(self, sdate, edate, size):
        date_list = mcal.get_calendar('NYSE').valid_days(start_date=sdate, end_date=edate)

        monthly_date_list = [date_list[0]]
        for idx in range(len(date_list)-1):
            today, tomorrow = date_list[idx], date_list[idx+1]
            if today.month != tomorrow.month:
                monthly_date_list.append(today.strftime('%Y-%m-%d'))
                
        try:
            print('Trying API call for universe')
            df = quandl.get_table('SHARADAR/DAILY',  
                                  date = monthly_date_list, 
                                  qopts={"columns":['date', 'marketcap','ticker']},
                                paginate=True).sort_values('marketcap', ascending=False
                            ).groupby('date').head(size).set_index('date').sort_index()
        except:
            print('Trying bulk download for universe')
            filename = DOWNLOAD_PATH+'/universe.zip'
            quandl.export_table('SHARADAR/DAILY',  
                                  date = monthly_date_list, 
                                  qopts={"columns":['date', 'marketcap','ticker']},
                                filename=filename)
            
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).sort_values('marketcap', ascending=False
                            ).groupby('date').head(size).set_index('date').sort_index()

        df = df.reset_index()

        return df
    
    @timeis
    def update_ticker(self, universe):
        try:
            print('Trying API call for tickers')
            df = quandl.get_table('SHARADAR/TICKERS',ticker=universe, table='SF1', 
                                     paginate=True).set_index('permaticker').sort_index()
        except:
            print('Trying bulk download for tickers')
            filename = DOWNLOAD_PATH+'/tickers.zip'
            quandl.export_table('SHARADAR/TICKERS',ticker=universe, table='SF1', filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('permaticker').sort_index()

        df = df.reset_index()

        return df
    
    @timeis
    def update_fundamentals(self, sdate, edate, universe):
        try:
            print('Trying API call for fundamentals')
            df = quandl.get_table('SHARADAR/SF1', 
                          ticker = universe,
                          datekey = {'gte':sdate,'lte':edate},
                          dimension = 'ART', paginate=True).set_index('datekey').sort_index()
        except:
            print('Trying bulk download for fundamentals')
            filename = DOWNLOAD_PATH+'/fundamentals.zip'
            quandl.export_table('SHARADAR/SF1', 
                          ticker = universe,
                          datekey = {'gte':sdate,'lte':edate},
                          dimension = 'ART', filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('datekey').sort_index()

        df = df.reset_index()

        return df
    
    @timeis
    def update_metric(self, sdate, edate, universe):
        try:
            print('Trying API call for metric')
            df = quandl.get_table('SHARADAR/DAILY', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, paginate=True
                          ).set_index('date').sort_index()
        except:
            print('Trying bulk download for metric')
            filename = DOWNLOAD_PATH+'/metric.zip'
            quandl.export_table('SHARADAR/DAILY', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('date').sort_index()
        
        df = df.reset_index()

        return df
    
    @timeis
    def update_market(self, sdate, edate, universe):
        try:
            print('Trying API call for market')
            df = quandl.get_table('SHARADAR/SEP', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, paginate=True).set_index('date').sort_index()
        except:
            print('Trying bulk download for market')
            filename = DOWNLOAD_PATH+'/market.zip'
            quandl.export_table('SHARADAR/SEP', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('date').sort_index()
        
        df = df.reset_index()

        return df
        
    @timeis
    def update_macro(self, fred_key, fred_list):

        macro_list = ['CPIAUCSL', 'PCE', 'M2', 'ICSA']
        macro_list = set(macro_list + fred_list)
        df = None
        for ticker in macro_list:
            df_add = self._get_PIT_df(ticker, fred_key)
            df_add['ticker'] = [ticker for _ in df_add.index]
            df = pd.concat([df, df_add],axis=0)

        df = df.reset_index()        
        return df

    def _get_PIT_df(self, ID, fred_key):
        API_KEY = fred_key
        REAL_TIME_START, REAL_TIME_END = '1900-01-01', '9999-12-31'
        
        url = 'https://api.stlouisfed.org/fred/series/observations?series_id={}'.format(ID)
        url += '&realtime_start={}&realtime_end={}&api_key={}&file_type=json'.format(
                                        REAL_TIME_START, REAL_TIME_END, API_KEY)
        
        response = requests.get(url)
        observations = json.loads(response.text)['observations']
        
        df = pd.DataFrame(observations).sort_values(['date','realtime_start']
            ).groupby('date').first()
        df.index = pd.to_datetime(df.index)
        df.realtime_start = pd.to_datetime(df.realtime_start)

        df['datekey'] = df.realtime_start
        df['is_inferred'] = (df.datekey == df.datekey.shift(1))|(
            df.datekey == df.datekey.shift(-1))

        non_inferred_df = df[df['is_inferred']==False]
        lag_list = [(y-x).days for x,y in 
                        zip(non_inferred_df.index, non_inferred_df.datekey)]
        mean_lag, max_lag = int(np.mean(lag_list)+1), int(np.max(lag_list)+1)
        
        df.datekey = [
            date + relativedelta(days=mean_lag) if df.loc[date].is_inferred
            else df.loc[date].datekey
            for date in df.index]

        df = df[['value','datekey','is_inferred']]
        df['cdate'] = df.index
        df = df.set_index('datekey')

        return df
    
    @timeis
    def update_index(self, yfinance_list):
        df = None
        yf_ticker_list = [
            '^GSPC','^IXIC','^DJI','^RUT','^VIX','^TNX','^SP500TR',
            'GC=F', 'CL=F']
        yf_ticker_list += [
            'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE',
            'XLK', 'XLU']
        yf_ticker_list += [
            'IVV', 'RWO', 'IDEV', 'IEMG', 'BND', 'EFA', 'VEA', 'EEM',
            'VNQ', 'SPY', 'DIA', 'GLD', 'IGF', 'USO', 'UUP', 'QQQ']

        yf_ticker_list = set(yf_ticker_list + yfinance_list)

        for ticker in yf_ticker_list:
            df_add = yf.Ticker(ticker).history(period='max')
            df_add.index.rename('date', inplace=True)
            df_add.columns = ['open', 'high', 'low', 'closeadj', 
                'volume', 'dividends', 'stock splits']
            df_add['ticker'] = [ticker for _ in df_add.index]
            df = pd.concat([df, df_add],axis=0)
            
        df = df.reset_index()
        
        return df
    
    @timeis
    def save_db(self, db_name, is_update):

        db = sqlite3.connect(db_name)

        if_exists = 'append' if is_update else 'replace'
        self.universe_df.to_sql(name='universe', con=db, if_exists=if_exists, index=False)
        self.ticker_df.to_sql(name='ticker', con=db, if_exists=if_exists, index=False)
        self.quanterly_report_df.to_sql(name='fundamentals', con=db, if_exists=if_exists, index=False)
        self.daily_metric_df.to_sql(name='metric', con=db, if_exists=if_exists, index=False)
        self.market_data_df.to_sql(name='market', con=db, if_exists=if_exists, index=False)
        self.macro_df.to_sql(name='macro', con=db, if_exists=if_exists, index=False)
        self.index_df.to_sql(name='indices', con=db, if_exists=if_exists, index=False)

        if is_update:
            def drop_duplicates(table, subset):
                qry = 'SELECT * FROM {}'.format(table)
                pd.read_sql(sql=qry, con=db).drop_duplicates(subset, keep='last'
                    ).sort_values(subset[-1]
                    ).to_sql(name=table, con=db, index=False, if_exists='replace')

            drop_duplicates('universe', ['ticker', 'date'],)
            drop_duplicates('ticker', ['ticker', 'permaticker'])
            drop_duplicates('fundamentals', ['ticker', 'datekey'])
            drop_duplicates('metric', ['ticker', 'date'])
            drop_duplicates('market', ['ticker', 'date'])
            drop_duplicates('macro', ['ticker', 'datekey'])
            drop_duplicates('indices', ['ticker', 'date'])


        date_list = pd.read_sql('SELECT * FROM market', db).date.sort_values().unique()

        print('DB has data from {} to {}'.format(
            date_list[0], date_list[-1]))

        db.close()
        
        print('Data saved in {}'.format(db_name))
