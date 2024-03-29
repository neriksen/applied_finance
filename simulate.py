import pandas as pd
import numpy as np
import arch
import datetime as dt
import scipy.stats as stats


class Market:
    def __init__(self, data_to_simulate, start_date = dt.date(2020, 1, 1), end_date = dt.date(2060, 12, 31)):
        
        # ---- Input data ----
        data_to_simulate.index = pd.to_datetime(data_to_simulate.index)
        self.data = data_to_simulate
        self.start = start_date
        self.end = end_date

        self.monthly = self.data.asfreq(freq='BMS', method='pad')
        self.yearly = self.data.asfreq(freq='BYS', method='pad')
        
        self.pct_change = self.data.pct_change()[1:]*100
        self.pct_monthly = self.monthly.pct_change()[1:]*100
        self.pct_yearly = self.yearly.pct_change()[1:]*100
        
        self.log_change = (np.log(self.data) - np.log(self.data.shift(1)))[1:]*100
        self.log_monthly = (np.log(self.monthly) - np.log(self.monthly.shift(1)))[1:]*100
        self.log_yearly = (np.log(self.yearly) - np.log(self.yearly.shift(1)))[1:]*100
        
        # ---- Simulation ----
        self.index = pd.date_range(start_date, end_date, freq='B')
        self.monthly_index = pd.date_range(start_date, end_date, freq='BMS')
        self.yearly_index = pd.date_range(start_date, end_date, freq='BYS')

        self.days = len(self.index)
        self.months = len(self.monthly_index)
        self.years = len(self.yearly_index)

        
    def norm_innovations(self, log = True, freq = 'D', random_state = None):
        '''
        Simulate a market using a fitted normal distribution
        
        log: Boolean (log change/pct change)
        freq: 'Y' (default), 'M', 'D'
        random_state = None (default), int

        returns Nx1 Dataframe
        '''
        data = pick_log_freq(self, log, freq)
        mean, var = stats.norm.fit(data)
        market_sample = stats.norm.rvs(mean, var, size= pick_horizon(self, freq), random_state = random_state)
        normalized = normalize_market(market_sample)
        
        return pd.DataFrame(normalized, index=pick_index(self, freq), columns=['Price'])        
        
        
    
    def t_innovations(self, log = True, freq = 'D', random_state = None, nc = False):
        '''
        Simulate a market using a fitted t distribution
        
        log: Boolean (log change/pct change)
        freq: 'Y' (default), 'M', 'D'
        random_state = None (default), int

        returns Nx1 Dataframe
        '''
        data = pick_log_freq(self, log, freq)
        
        if nc:
            df, nc, loc, scale = stats.nct.fit(data)
            market_sample = stats.nct.rvs(df, nc, loc, scale, size= pick_horizon(self, freq), random_state = random_state)
        else:
            df, loc, scale = stats.t.fit(data)
            market_sample = stats.t.rvs(df, loc, scale, size= pick_horizon(self, freq), random_state = random_state)        
    
        normalized = normalize_market(market_sample)
        
        return pd.DataFrame(normalized, index=pick_index(self, freq), columns=['Price'])

    
    
    def draw(self, log = True, freq = 'D', with_replacement = True, random_state = None):
        '''
        Simulated a market using drawing

        freq: 'Y' (default), 'M', 'D'
        with_replacement: Boolean
        random_state = None (default), int

        returns Nx1 Dataframe
        '''
        data = pick_log_freq(self, log, freq)

        market_data = data.sample(n = pick_horizon(self, freq), replace=with_replacement, random_state = random_state)
        normalized = normalize_market(market_data)
        
        return pd.DataFrame(normalized, index=pick_index(self, freq), columns=['Price'])
    
    
    def garch(self, p=1, o=1, q=1, random_state = None, log = True, mu_override = ""):
        '''
        Simulated a market using a gjr-garch(1,1) with a skewed t-distribution to draw error term

        returns

        a dataframe with returns, volatility, error terms and market prices given index 100 at t=0
        '''
        data = pick_log_freq(self, log, freq = 'D')
        
        dist = arch.univariate.SkewStudent(np.random.RandomState(random_state))
        vol = arch.univariate.GARCH(p=p, o=o, q=q)
        model = arch.univariate.ConstantMean(data, volatility=vol, distribution=dist)
        results = model.fit(disp = 'off')
        params = results.params
        
        if not mu_override == "":
            params['mu'] = mu_override
        
        returns = model.simulate(params, pick_horizon(self, freq = 'D'))
        
        #Ensure returns cannot dip below -100%
        returns['data'] = returns['data'].apply(lambda x: max(-99.9, x))
        
        returns['Price'] = normalize_market(returns['data'].values)
        
        returns.index = pick_index(self, freq = 'D')
        
        return returns


    
def pick_index(market, freq):
    if freq == 'D':
        return market.index
    elif freq == 'M':
        return market.monthly_index
    elif freq == 'Y':
        return market.yearly_index
    
    
def pick_horizon(market, freq):
    if freq == 'D':
        return market.days
    elif freq == 'M':
        return market.months
    elif freq == 'Y':
        return market.years
    
    
def pick_log_freq(market, log, freq):
    if log:
        if freq == 'D':
            return market.log_change
        elif freq == 'M':
            return market.log_monthly
        else:
            return market.log_yearly
    else:
        if freq == 'D':
            return market.pct_change
        elif freq == 'M':
            return market.pct_monthly
        else:
            return market.pct_yearly

        
def normalize_market(market_data):
    '''
    Normalize returns formatted as percent
    '''
    horizon = len(market_data)
    market = np.empty((horizon, 1))
    market[0, 0] = 100
    market_data.tolist()
    #for i, returns in enumerate(market_data, start=1):
    #    if i < (horizon):
    #        market[i, 0] = market[i-1, 0]*(1+returns/100)
    market_data /= 100        
    for i in range(1, horizon):
        market[i, 0] = market[i-1, 0]*(1+market_data[i])
        
    return market


