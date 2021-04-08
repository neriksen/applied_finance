import numpy as np
import pandas as pd
import time
from itertools import product
from multiprocessing.pool import Pool
import datetime as dt
import simulate

import matplotlib.pyplot as plt

import calculate_return as cr

def summary_stats(returns, values, h, annual_rf):
    mean = ((values[-1] / values[0]) ** (1 / h) - 1) * 3.46410
    std = returns.std() * 3.46410  # sqrt(12) = 3.46410
    return np.array([mean, std, (mean - annual_rf) / std, values[-1]]).transpose()



if __name__ == "__main__":

    #profiler = cProfile.Profile()
    #profiler.enable()

    # Creating returns to simulate
    spx = pd.read_csv('^GSPC.csv', index_col=0)
    savings_year = pd.read_csv('investment_plan_year.csv', sep=';', index_col=0)
    savings_year.index = pd.to_datetime(savings_year.index, format='%Y')
    savings_month = (savings_year.resample('BMS').pad() / 12)['Earnings'].values

    YEARLY_RF = 0.015
    YEARLY_RM = 0.04  # Weighted average of margin rates

    # --- Fixed parameters ----
    investments = savings_month * 0.05
    #investments = np.load('sims/default_settings/investments.npy')
    #np.save('sims/default_settings/investments.npy', investments)

    START = dt.date(2020, 1, 1)
    END = dt.date(2080, 1, 31)
    Market = simulate.Market(spx.iloc[-7500:, -2], START, END)

    GAMMA = 2.5
    #SIGMA = Market.yearly.pct_change().std() ** 2
    SIGMA = 0.028372570500884393
    #MR = (Market.yearly[-1] / Market.yearly[0]) ** (1 / len(Market.yearly)) - 1
    MR = 0.07601121293459889
    COST = 0.002
    # --- End fixed parameters ----

    # Creating list of arguments
    a = [[investments], ['garch'], range(1000), [1],
         [GAMMA], [SIGMA], [MR], [0.02], [0.05], [COST]]

    #main(investments_in, sim_type, random_state, gearing_cap, gamma, sigma, mr,
    #     yearly_rf, yearly_rm, cost)

    comb_args = tuple(product(*a))

    #arg_iter = (i for i in comb_args)
    #num_sims = sum(1 for _ in arg_iter)
    #print('number of simulations to run: ', num_sims)


    with Pool() as p:
        tic = time.perf_counter()
        res = p.starmap(cr.main, comb_args, 2)
        toc = time.perf_counter()
        dfs = pd.concat(res)
        print(f"Script took {toc - tic:0.5f} seconds")
        print(dfs)

    #profiler.disable()
    #stats = pstats.Stats(profiler)
    #stats.strip_dirs()
    #stats.sort_stats('ncalls')
    #stats.reverse_order()
    #stats.print_stats()
