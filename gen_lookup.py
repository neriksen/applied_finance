import simulate, pandas as pd, datetime as dt, numpy as np
from multiprocessing import Pool


def sim_market_garch(ran_state, Market):
    market = (Market.garch(log=False, random_state=ran_state, mu_override=0.030800266141550736)
              .asfreq('BMS', 'pad')['Price']
                      .pct_change()
                      .values)
    np.save('market_lookup/garch/' + str(ran_state) + '.npy', market)
    print(ran_state)

    
def sim_market_norm(ran_state, Market):
    market = (Market.norm_innovations(random_state = ran_state, freq='M').asfreq('BMS', 'pad')['Price']
                      .pct_change()
                      .values)
    np.save('market_lookup/norm/' + str(ran_state) + '.npy', market)
    print(ran_state)
    
    
def sim_market_t(ran_state, Market):
    market = (Market.t_innovations(random_state = ran_state, freq='M')
              .asfreq('BMS', 'pad')['Price']
                      .pct_change()
                      .values)
    np.save('market_lookup/t/' + str(ran_state) + '.npy', market)
    print(ran_state)
    
    
def sim_market_draw(ran_state, Market):
    market = (Market.draw(random_state = ran_state, freq='M').asfreq('BMS', 'pad')['Price']
                      .pct_change()
                      .values)
    np.save('market_lookup/draw/' + str(ran_state) + '.npy', market)
    print(ran_state)
    

if __name__ == '__main__':
    # Creating returns to simulate
    spx = pd.read_csv('^GSPC.csv', index_col=0)
    START = dt.date(2020, 1, 1)
    END = dt.date(2080, 1, 31)
    Market = simulate.Market(spx.iloc[-7500:, -2], START, END)
    
    from itertools import product
    
    # Creating list of arguments
    a = [range(10000), [Market]]
    states = tuple(product(*a))

    with Pool() as p:
        for sim in [sim_market_garch, sim_market_t, sim_market_draw, sim_market_norm]:
            p.starmap(sim, states)
