import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import arch
import datetime as dt


def simulate_market(years, data_to_simulate, p=1, o=0, q=1):
    '''
    Simulated a market using a gjr-garch(1,1) with a skewed t-distribution to draw error term
    
    returns
    
    a dataframe with returns, volatility, error terms and market prices given index 100 at t=0
    '''
    model=arch.arch_model(data_to_simulate, vol='Garch', p=p, o=o, q=q, dist='skewt')
    results=model.fit(disp='off')
    
    #setting horizon
    horizon = 252*years

    #rs = np.random.RandomState()
    #state = rs.get_state()

    #dist = arch.univariate.SkewStudent(random_state=rs)
    dist = arch.univariate.SkewStudent()
    vol = arch.univariate.GARCH(p=p, o=o, q=q)
    repro_mod = arch.univariate.ConstantMean(None, volatility=vol, distribution=dist)

    returns=repro_mod.simulate(results.params, horizon)
    
    market = np.empty((horizon, 1))
    market[0, 0] = 100
    for i, err in enumerate(returns["data"].values, start=1):
        if i < (horizon):
            market[i, 0] = market[i-1, 0]*(1+err/100)
    
    returns['Price'] = pd.DataFrame({'Price': market[:, 0]})
    
    return returns
