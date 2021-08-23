import numpy as np
import pandas as pd

class Strategy():
    def __init__(self):
        self.date = None
        self.cache = None 

    def compute_target(self, universe_list):
        target_weight = {}
        for ticker in universe_list:
            target_weight[ticker] = 1
            
        target_weight = self.normalize(target_weight)

        return target_weight
    
    def default_factor(self, ticker, ftype):
        if ftype == 'GP/A':
            gp = self.get_value(ticker, 'fundamentals','gp')
            assets = self.get_value(ticker, 'fundamentals','assets')
            return gp/assets
        
        elif ftype == 'RND/A':
            rnd = self.get_value(ticker, 'fundamentals','rnd')
            assets = self.get_value(ticker, 'fundamentals','assets')
            return rnd/assets
        
        elif ftype == 'E/M':
            pe = self.get_value(ticker,'metric','pe')
            return 1/pe
        
        elif ftype == 'momentum':
            p_1M = self.get_value(ticker,'metric','closeadj',20)
            p_6M = self.get_value(ticker,'metric','closeadj',120)
            return np.log(p_1M/p_6M)
        
        elif ftype == 'mean_reversion':
            p_now = self.get_value(ticker,'metric','closeadj')
            p_1M = self.get_value(ticker,'metric','closeadj',1)
            return np.log(p_1M/p_now)
        
        elif ftype == 'EPSG/PE':
            pe = self.get_value(ticker,'metric','pe')
            eps_growth = np.log(self.get_value(ticker,'fundamentals','eps')/self.get_value(ticker,'fundamentals','eps',1))
            return eps_growth/pe
        
        elif ftype == 'inv_vol':
            vol = self.get_value_list(ticker,'fundamentals','closeadj', 120).apply(np.log).diff().std()
            return 1/vol
        
        elif ftype == 'CAPM_alpha':
            market_returns = self.get_value_list('S&P500','index','closeadj',120).apply(np.log).diff()[1:]
            market_returns = np.array(market_returns).reshape(-1,1)
            stock_returns = self.get_value_list(ticker,'market','closeadj',120).apply(np.log).diff()[1:]
            regr = linear_model.LinearRegression()
            regr.fit(market_returns, stock_returns)
            beta, alpha = regr.coef_[0], regr.intercept_
            return alpha

        elif ftype == 'CAPM_return':
            market_returns_full = self.get_value_list('S&P500','index','closeadj').apply(np.log).diff().resampel('M').mean().shift()
            stock_returns = self.get_value_list(ticker,'market','closeadj').apply(np.log).diff().resampel('M').mean()

            market_returns = market_returns_full.loc[stock_returns.index]
            market_returns = np.array(market_returns).reshape(-1,1)

            regr = linear_model.LinearRegression()
            regr.fit(market_returns, stock_returns)
            beta, alpha = regr.coef_[0], regr.intercept_

            prediction = alpha + beta*market_returns_full.iloc[-1]

            return prediction
        
        elif ftype == 'F-score':
            ROA = self.get_value(ticker,'fundamentals','roa')
            ROA_1Y = self.get_value(ticker,'fundamentals','roa', 4)
            OCF = self.get_value(ticker,'fundamentals','ncfo')
            ASSET = self.get_value(ticker,'fundamentals','assets')
            
            DE = self.get_value(ticker,'fundamentals','de')
            DE_1Y = self.get_value(ticker,'fundamentals','de', 4)
            CR = self.get_value(ticker,'fundamentals','currentratio')
            CR_1Y = self.get_value(ticker,'fundamentals','currentratio',4)
            SHARE = self.get_value(ticker,'fundamentals','sharesbas')
            SHARE_1Y = self.get_value(ticker,'fundamentals','sharesbas', 4)
            
            GM = self.get_value(ticker,'fundamentals','grossmargin')
            GM_1Y = self.get_value(ticker,'fundamentals','grossmargin', 4)
            AT = self.get_value(ticker,'fundamentals','assetturnover')
            AT_1Y = self.get_value(ticker,'fundamentals','assetturnover', 4)
            
            point = 1.*(ROA > 0) + 1.*(OCF > 0) + 1.*(ROA > ROA_1Y) + 1.*(OCF/ASSET > ROA)
            point += 1.*(DE < DE_1Y) + 1.*(CR > CR_1Y) + 1.*(SHARE < SHARE_1Y)
            point += 1.*(GM > GM_1Y) + 1.*(AT > AT_1Y)
            
            return point
            
        else:
            assert False

    def custom_factor(self, ticker, ftype):
        assert False
    
    def compute_factor(self, ticker, ftype):
        try:
            factor = self.default_factor(ticker, ftype)
        except:
            factor = self.custom_factor(ticker, ftype)
        return factor
    
    def compute_factor_zscore(self, universe_list, ftype, groupby_sector=False):
        ticker_to_factor = {}
        ticker_to_sector = {}
        for ticker in universe_list:
            try:
                factor = self.compute_factor(ticker, ftype)
                sector = self.get_value(ticker,'tickerinfo','sector')
            except:
                factor = np.nan
            if np.isnan(factor):
                pass
            else:
                ticker_to_factor[ticker] = factor
                ticker_to_sector[ticker] = sector
        factor_series = pd.Series(ticker_to_factor).dropna().sort_values(ascending=False)
        sector_series = pd.Series(ticker_to_sector).dropna()
        
        
        if groupby_sector:
            df = pd.concat([factor_series, sector_series], axis=1).dropna()
            df.columns=['factor','sector']
            df = df.groupby('sector').transform(lambda x:(x-x.mean())/x.std(ddof=0)).dropna()
            factor_series = df['factor'].sort_values(ascending=False)
        else:
            factor_series = (factor_series-factor_series.mean())/factor_series.std()
        
        return factor_series
    
    def normalize(self, target_weight):
        target_sum = sum([np.abs(x) for x in target_weight.values()])
        target_weight = {ticker:target_weight[ticker]/target_sum for ticker in target_weight}
        assert np.abs(sum(target_weight.values())-1) < 1e-6
        return target_weight
    
    def get_value(self, ticker, table, value, lag=0):
        try:
            if table == 'tickerinfo':
                x = self.cache[table][ticker][value].iloc[0]
            else:
                x = self.cache[table][ticker][value].iloc[-1-lag]
        except:
            x = np.nan
        return x
    
    def get_value_list(self, ticker, table, value, lag='max'):
        try:
            if lag == 'max':
                x = self.cache[table][ticker][value]
            else:
                x = self.cache[table][ticker][value].iloc[-1-lag:]
        except:
            x = np.nan
        return x