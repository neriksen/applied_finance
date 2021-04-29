"""Optimised version of strategy.py to calculate performance of
a leveraged investment strategy"""

import math
import time
from multiprocessing.pool import Pool
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from debt import Debt


debt_available = {'SU': Debt(), 'Nordnet': Debt()}


def determine_investment(phase, pv_u, tv_u, s, td, pi_rf, dst, g, period):
    # returns cash, new_equity, new_debt

    if phase == 1:
        # Check if gearing cap has been reached
        equity = tv_u + s - td
        if td > (equity * g):
            new_debt = 0
        else:
            new_debt = nd(g, s, tv_u, td, dst, period)
        return 0, s, new_debt

    if phase == 2:
        stocks_sold = max(pv_u - dst, 0)
        debt_repayment = min(td, s + stocks_sold)
        repayment_left = debt_repayment

        try:
            repayment = min(debt_repayment, debt_available['Nordnet'].debt_amount)
            debt_available['Nordnet'].prepayment(repayment)
            repayment_left = debt_repayment - repayment
        except KeyError:
            pass

        try:
            debt_available['SU'].prepayment(repayment_left)
        except KeyError:
            pass

        leftover_savings = max(s - debt_repayment - stocks_sold, 0)
        return 0, leftover_savings, -debt_repayment

    if phase == 3:
        return 0, s, 0

    if phase == 4:
        desired_cash = (1 - pi_rf) * (tv_u + s)
        desired_savings = (pi_rf) * (tv_u + s)
        change_in_stock = desired_savings - pv_u
        return desired_cash, change_in_stock, 0


# Function assumes monthly periods
def nd(g, s, tv_u, td, dst, period):
    equity = tv_u + s - td
    total_desired_debt = min(g / (g + 1) * dst, equity * g)
    remaining_debt_needed = max(0, total_desired_debt - td)

    SU_amount, Nordnet_amount = 0, 0

    if period <= 60:
        try:
            # Has SU already been taken?
            SU_amount = min(3234, remaining_debt_needed)
            debt_available['SU'].add_debt(SU_amount)

            remaining_debt_needed -= SU_amount
        except KeyError:
            pass

    try:
        # Has Nordnet already been taken?
        Nordnet_amount = min(max(0, g * equity), remaining_debt_needed)
        debt_available['Nordnet'].add_debt(Nordnet_amount)
    except KeyError:
        pass

    return SU_amount + Nordnet_amount


def interest_all_debt(period):
    interest_bill = 0
    try:
        if period <= 60:
            # No deduction for SU debt while studying
            interest_bill += debt_available['SU'].calculate_interest(deduction=0)
        else:
            interest_bill += debt_available['SU'].calculate_interest()
    except KeyError:
        pass

    try:
        interest_bill += debt_available['Nordnet'].calculate_interest(deduction=0)
    except KeyError:
        pass

    return interest_bill


def phase_check(phase, pi_rf, pi_rm, pi_hat, td, dual_phase):
    if phase == 4:
        return 4

    if td > 0:
        # has target not been reached?
        if pi_hat < pi_rm and phase <= 1:
            return 1
        # if target has been reached once and debt remains, stay in phase 2
        return 2

    # if target has been reached and no debt remains
    # is the value still above the target?
    if pi_hat < pi_rf and dual_phase:
        return 3
    return 4


def pi_arr(rate, gamma, sigma2, mr, cost):
    return max(0, ((mr - cost - rate) / (gamma * sigma2)))


def calc_pi(gamma, sigma2, mr, rate, cost=0.0):
    # Assumes rate is a np.array
    pi_vec = np.vectorize(pi_arr)
    pi = np.apply_along_axis(pi_vec, 0, rate, gamma, sigma2, mr, cost)
    return pi


def calculate_return(savings_in, returns, gearing_cap, pi_rf_in, pi_rm_in, rf_in, rm_in, pay_taxes, dual_phase):


    # Setting up constants and dataframe for calculation
    ses_val = savings_in.sum()  # Possibly add more sophisticated discounting
    ist = pi_rm_in[0] * ses_val
    columns = ['period', 'savings', 'cash', 'new_equity', 'new_debt', 'total_debt', 'nip', 'pv_p',
               'interest', 'market_returns', 'pv_u', 'tv_u', 'equity', 'dst', 'phase', 'pi_hat',
               'g_hat', 'SU_debt', 'Nordnet_debt', 'rf', 'rm', 'pi_rf', 'pi_rm']

    len_savings = len(savings_in)
    len_columns = len(columns)

    pp = np.zeros((len_savings, len_columns))

    period, savings, cash, new_equity, new_debt, total_debt, nip, pv_p, interest, \
    market_returns, pv_u, tv_u, equity, dst, phase, pi_hat, g_hat, SU_debt, Nordnet_debt, rf, rm,\
    pi_rf, pi_rm = range(len_columns)

    tax_deduction = 0

    pp[:, period] = range(len_savings)
    pp[:, market_returns] = returns
    pp[:, savings] = savings_in
    pp[:, rf] = rf_in
    pp[:, rm] = rm_in
    pp[:, pi_rf] = pi_rf_in
    pp[:, pi_rm] = pi_rm_in
    pp[0, market_returns] = 0

    debt_pct_offset = np.round(rm_in - 0.001918504646, 3)
    # Initializing debt objects
    try:
        debt_available['SU'] = Debt(rate_structure=[[0, 0, 0.04 + debt_pct_offset[0]]],
                                    rate_structure_type='relative', initial_debt=0)
    except KeyError:
        pass

    try:
        debt_available['Nordnet'] = Debt(rate_structure=[[0, .4, 0.02 + debt_pct_offset[0]],
                                                         [.4, .6, 0.03 + debt_pct_offset[0]],
                                                         [.6, 0, 0.07] + debt_pct_offset[0]],
                                         rate_structure_type='relative', initial_debt=0)
    except KeyError:
        pass

    # Period 0 primo
    pp[0, cash] = 0
    pp[0, new_equity] = pp[0, savings]
    pp[0, new_debt] = pp[0, new_equity] * gearing_cap
    pp[0, total_debt] = pp[0, new_debt]
    pp[0, SU_debt] = min(pp[0, new_debt], 3248)
    pp[0, Nordnet_debt] = max(0, pp[0, new_debt] - 3248)

    # Adding debt to SU and Nordnet objects
    try:
        debt_available['SU'].add_debt(pp[0, SU_debt])
    except KeyError: pass
    try:
        debt_available['Nordnet'].add_debt(pp[0, Nordnet_debt])
    except KeyError: pass

    pp[0, nip] = pp[0, new_debt] + pp[0, new_equity]
    pp[0, pv_p] = pp[0, nip]
    pp[0, pi_hat] = pp[0, pv_p] / ses_val

    # Period 0 ultimo
    pp[0, interest] = max(interest_all_debt(period=0), 0)
    pp[0, pv_u] = pp[0, pv_p]
    pp[0, tv_u] = pp[0, pv_u] + pp[0, cash]
    pp[0, equity] = pp[0, tv_u] - pp[0, total_debt]
    pp[0, dst] = ist
    pp[0, phase] = 1

    # Looping over all remaining periods
    for i in range(1, len_savings):

        # Period t > 0 primo
        if not (pp[i - 1, tv_u] <= 0 and (pp[i - 1, interest] > pp[i, savings])):

            pp[i, cash] = pp[i - 1, cash] * (1 + pp[i, rf]*(1-0.42))
            pp[i, cash], pp[i, new_equity], pp[i, new_debt] = determine_investment(
                pp[i - 1, phase], pp[i - 1, pv_u],
                pp[i - 1, tv_u], pp[i, savings], pp[i - 1, total_debt],
                pp[i - 1, pi_rf], pp[i - 1, dst], gearing_cap, pp[i, period])
            try:
                pp[i, SU_debt] = debt_available['SU'].debt_amount
                pp[i, Nordnet_debt] = debt_available['Nordnet'].debt_amount
            except KeyError:
                pass

            pp[i, total_debt] = pp[i - 1, total_debt] + pp[i, new_debt]
            pp[i, nip] = pp[i, new_equity] + max(0, pp[i, new_debt])
            pp[i, pv_p] = pp[i - 1, pv_u] + pp[i, nip]

            # Update debt cost
            try:
                if period <= 60:
                    debt_available['SU'].change_rate_structure([[0, 0, 0.04 + debt_pct_offset[i]]], 'relative')
                else:
                    debt_available['SU'].change_rate_structure([[0, 0, 0.01 + debt_pct_offset[i]]], 'relative')
            except KeyError: pass

            try:
                debt_available['Nordnet'].change_rate_structure([[0, .4, 0.02 + debt_pct_offset[i]],
                                                                 [.4, .6, 0.03 + debt_pct_offset[i]],
                                                                 [.6, 0, 0.07 + debt_pct_offset[i]]], 'relative')
            except KeyError: pass

            # Period t > 0 ultimo
            #if pp[i, period] == 60 and 'SU' in debt_available.keys():
            #    debt_available['SU'].change_rate_structure([[0, 0, 0.01]], 'dollar')

            pp[i, interest] = max(interest_all_debt(pp[i, period]), 0)
            pp[i, pv_u] = pp[i, pv_p] * (1 + pp[i, market_returns])

            # Check if we are in december to calculate taxes
            if pay_taxes and pp[i, period] % 12 == 0:
                year_return = pp[i, pv_u]-pp[i-12, pv_p]

                if year_return >= 0:  # Case we earned money
                    tax_base = max(0, year_return - tax_deduction)
                    tax_bill = min(56600, tax_base)*0.27 + max(0, (tax_base-56600))*0.42

                    # Deduct tax bill from portfolio value
                    pp[i, pv_u] -= tax_bill

                    # Update remaining tax deduction if any
                    tax_deduction -= min(tax_deduction, year_return)

                else:                  # Case we lost money
                    # Update tax deduction
                    tax_deduction += max(0, -year_return)

            pp[i, pv_u] -= pp[i, interest]

            pp[i, tv_u] = pp[i, pv_u] + pp[i, cash]
            pp[i, equity] = pp[i, tv_u] - pp[i, total_debt]
            pp[i, pi_hat] = min(pp[i, pv_u] / ses_val, pp[i, pv_u] / pp[i, tv_u])
            pp[i, phase] = phase_check(pp[i - 1, phase], pp[i-1, pi_rf], pp[i-1, pi_rm], pp[i, pi_hat], pp[i, total_debt], dual_phase)
            target_pi = pp[i-1, pi_rm] if pp[i - 1, phase] < 3 else pp[i-1, pi_rf]
            pp[i, dst] = max(pp[i, tv_u] * target_pi, ist)  # Moving stock target
            # pp[i, dst] = max(pp[i-1, dst], max(pp[i, tv_u]*target_pi, ist))  # Locked stock target at highest previous position

        else:
            #print('warning: catastrophic wipeout')
            pp[i:, [savings, cash, new_equity, new_debt, nip, pv_p,
                    interest, pv_u, tv_u, pi_hat, g_hat]] = 0

            cols = [total_debt, SU_debt, Nordnet_debt, equity, dst, phase]
            pp[i:, cols] = pp[i - 1, cols]

            break

    pp[:, g_hat] = pp[:, total_debt] / pp[:, equity]
    pp = pd.DataFrame(pp, columns=columns)
    return pp


def calculate100return(savings_in, returns, pay_taxes):
    # Running controls
    len_savings = len(savings_in)
    #assert len_savings == len(returns), 'Investment plan should be same no of periods as market'

    columns = ['period', 'savings', 'pv_p', 'market_returns', 'tv_u']

    pp = np.empty((len_savings, len(columns)))

    period, savings, pv_p, market_returns, tv_u = range(5)

    tax_deduction = 0

    pp[:, period] = range(len_savings)
    pp[:, market_returns] = returns
    pp[:, savings] = savings_in
    pp[0, market_returns] = 0
    pp[0, pv_p] = pp[0, savings]
    pp[0, tv_u] = pp[0, savings]

    for i in range(1, len_savings):
        # Period t > 0 primo
        pp[i, pv_p] = pp[i - 1, tv_u] + pp[i, savings]

        # Period t > 0 ultimo
        pp[i, tv_u] = pp[i, pv_p] * (1 + pp[i, market_returns])

        # Check if we are in december to calculate taxes
        if pay_taxes and pp[i, period] % 12 == 0:
            year_return = pp[i, tv_u] - pp[i - 12, pv_p]

            if year_return >= 0:  # Case we earned money
                tax_base = max(0, year_return - tax_deduction)
                tax_bill = min(56600, tax_base) * 0.27 + max(0, (tax_base - 56600)) * 0.42

                # Deduct tax bill from portfolio value
                pp[i, tv_u] -= tax_bill

                # Update remaining tax deduction if any
                tax_deduction -= min(tax_deduction, year_return)

            else:  # Case we lost money
                # Update tax deduction
                tax_deduction += max(0, -year_return)

    pp = pd.DataFrame(pp, columns=columns)
    return pp


def calculate9050return(savings_in, returns, rf_in, pay_taxes):
    # Strategy where 90% of value is initially invested in stocks, rest in risk free asset
    # Ratio of stocks falls linearly to 50% by age 65 and stays there

    # Running controls
    len_savings = len(savings_in)
    #assert len_savings == len(returns), 'Investment plan should be same no of periods as market'

    columns = ['period', 'savings', 'cash', 'pv_p', 'market_returns', 'pv_u', 'tv_u', 'ratio', 'rf']
    len_columns = len(columns)

    pp = np.empty((len_savings, len_columns))

    period, savings, cash, pv_p, market_returns, pv_u, tv_u, ratio, rf = range(len_columns)

    tax_deduction = 0

    pp[:, period] = range(len_savings)
    pp[:, market_returns] = returns
    pp[:, savings] = savings_in
    pp[:, rf] = rf_in
    pp[0, market_returns] = 0
    pp[0, pv_p] = pp[0, savings] * 0.9
    pp[0, cash] = pp[0, savings] * 0.1
    pp[0, pv_u] = pp[0, pv_p]
    pp[0, tv_u] = pp[0, savings]
    pp[0, ratio] = 90

    for i in range(1, len_savings):
        ratio_val = max(90 - pp[i, period] / 12, 50)
        pp[i, ratio] = ratio_val

        # Period t > 0 primo
        pp[i, pv_p] = pp[i - 1, pv_u] + pp[i, savings] * (ratio_val / 100)
        pp[i, cash] = pp[i - 1, cash] * (1 + pp[i, rf]*(1-0.42)) + pp[i, savings] * (1 - ratio_val / 100)

        # Period t > 0 ultimo
        pp[i, pv_u] = pp[i, pv_p] * (1 + pp[i, market_returns])

        # Check if we are in december to calculate taxes
        if pay_taxes and pp[i, period] % 12 == 0:
            year_return = pp[i, pv_u] - pp[i - 12, pv_p]

            if year_return >= 0:  # Case we earned money
                tax_base = max(0, year_return - tax_deduction)
                tax_bill = min(56600, tax_base) * 0.27 + max(0, (tax_base - 56600)) * 0.42

                # Deduct tax bill from portfolio value
                pp[i, pv_u] -= tax_bill

                # Update remaining tax deduction if any
                tax_deduction -= min(tax_deduction, year_return)

            else:  # Case we lost money
                # Update tax deduction
                tax_deduction += max(0, -year_return)

        pp[i, tv_u] = pp[i, pv_u] + pp[i, cash]

    pp = pd.DataFrame(pp, columns=columns)

    return pp


def main(investments_in, sim_type, random_state, gearing_cap,
         rf, rm, pi_rm, pi_rf, pay_taxes = True,
         seed_index=True, cost = 0.002):

    returns = np.load('market_lookup/' + sim_type + '/' + str(random_state) + '.npy')[0:len(investments_in)]

    returns -= cost/12

    port = calculate_return(investments_in, returns, gearing_cap, pi_rf, pi_rm, rf, rm,
                            pay_taxes, dual_phase = True)
    port_single = calculate_return(investments_in, returns, gearing_cap, pi_rm, pi_rm, rf, rm,
                                   pay_taxes, dual_phase=False)
    port100 = calculate100return(investments_in, returns, pay_taxes)
    port9050 = calculate9050return(investments_in, returns, rf, pay_taxes)

    # Joining normal strategies on to geared
    port['dual_phase'] = port['tv_u'] - port['total_debt']
    port['single_phase'] = port_single['tv_u'] - port_single['total_debt']
    port['100'] = port100['tv_u']
    port['9050'] = port9050['tv_u']
    port['random_state'] = random_state

    # Convert selected float columns to integer values
    flt_cols = ['period', 'random_state', 'savings', 'cash', 'new_equity', 'new_debt', 'total_debt',
                'pv_p', 'interest', 'tv_u', 'dst', 'phase', '100', '9050']

    port.loc[:, flt_cols] = port.loc[:, flt_cols].astype(int)
    
    # Reducing size of port
    # Setting period as index
    if seed_index:
        port.set_index(['random_state', 'period'], drop=True, inplace=True)
    else:
        port.set_index('period', drop=True, inplace=True)

    return port


def main_shiller(investments_in, returns, rf, rm, pi_rf, pi_rm, gearing_cap = 1, pay_taxes=True):
    
    port = calculate_return(investments_in, returns, gearing_cap, pi_rf, pi_rm, rf, rm,
                            pay_taxes, dual_phase=True)
    port_single = calculate_return(investments_in, returns, gearing_cap, pi_rf, pi_rm, rf, rm,
                                   pay_taxes, dual_phase=False)
    port100 = calculate100return(investments_in, returns, pay_taxes)
    port9050 = calculate9050return(investments_in, returns, rf, pay_taxes)

    # Joining normal strategies on to geared
    port['dual_phase'] = port['tv_u'] - port['total_debt']
    port['single_phase'] = port_single['tv_u'] - port_single['total_debt']
    port['100'] = port100['tv_u']
    port['9050'] = port9050['tv_u']

    # Convert selected float columns to integer values
    flt_cols = ['period', 'savings', 'cash', 'new_equity', 'new_debt', 'total_debt',
                'pv_p', 'interest', 'tv_u', 'dst', 'phase', '100', '9050']

    port.loc[:, flt_cols] = port.loc[:, flt_cols].astype(int)

    # Reducing size of port
    # Setting period as index
    port.set_index('period', drop=True, inplace=True)

    return port


def fetch_returns_shiller(returns, YEARLY_RF, YEARLY_RM, BEGINNING_SAVINGS=9000, YEARLY_INCOME_GROWTH=0.03,
                          PAY_TAXES=True, YEARS=50, GAMMA=2, COST=0.002, SIGMA2=0.02837, MR=0.076, **kwargs):

    SLOPE = (0.014885 + YEARLY_INCOME_GROWTH / 12) * BEGINNING_SAVINGS
    CONVEXITY = -0.0000373649 * BEGINNING_SAVINGS
    JERK = 0.000000025 * BEGINNING_SAVINGS
    savings_func = lambda x: JERK * (x ** 3) + CONVEXITY * (x ** 2) + SLOPE * x + BEGINNING_SAVINGS

    # In case RF or RM is inputted as constants convert to numpy arrays
    if not isinstance(YEARLY_RF, np.ndarray):
        YEARLY_RF = np.full(len(returns), YEARLY_RF)

    if not isinstance(YEARLY_RM, np.ndarray):
        YEARLY_RM = np.full(len(returns), YEARLY_RM)
    
    if not 'PI_RF' in kwargs:
        PI_RF = calc_pi(GAMMA, SIGMA2, MR, YEARLY_RF, COST)
    else:
        PI_RF = kwargs['PI_RF']
    if not 'PI_RM' in kwargs:
        PI_RM = calc_pi(GAMMA, SIGMA2, MR, YEARLY_RM, COST)
    else:
        PI_RM = kwargs['PI_RM']
    
    # Converting RF and RM to monthly rates
    RM = np.exp(YEARLY_RM/12) -1
    RF = np.exp(YEARLY_RF/12) -1

    savings_val = np.array([savings_func(x) for x in range(0, YEARS * 12 + 1)])
    investments = savings_val * 0.05

    assert(len(investments) == len(returns))

    # Deduct yearly cost from returns
    returns -= COST/12

    res = main_shiller(investments, returns, RF, RM, PI_RF, PI_RM, 1, PAY_TAXES)

    return res


def fetch_returns(sim_type, random_seeds, BEGINNING_SAVINGS = 9000,
                   YEARLY_INCOME_GROWTH = 0.03, PAY_TAXES = True, YEARS = 50, GAMMA = 2,
                   YEARLY_RF = 0.02, YEARLY_RM = 0.023, COST = 0.002,
                   SIGMA2 = 0.02837, MR = 0.076, SEED_INDEX = True):

    SLOPE = (0.014885 + YEARLY_INCOME_GROWTH/12) * BEGINNING_SAVINGS
    CONVEXITY = -0.0000373649 * BEGINNING_SAVINGS
    JERK = 0.000000025 * BEGINNING_SAVINGS
    savings_func = lambda x: JERK * (x ** 3) + CONVEXITY * (x ** 2) + SLOPE * x + BEGINNING_SAVINGS
        
    savings_val = np.array([savings_func(x) for x in range(0, YEARS*12 + 1)])
    investments = savings_val * 0.05

    # In case RF or RM is inputted as constants convert to numpy arrays
    if not isinstance(YEARLY_RF, np.ndarray):
        YEARLY_RF = np.full(len(investments), YEARLY_RF)

    if not isinstance(YEARLY_RM, np.ndarray):
        YEARLY_RM = np.full(len(investments), YEARLY_RM)

    PI_RF = calc_pi(GAMMA, SIGMA2, MR, YEARLY_RF, COST)
    PI_RM = calc_pi(GAMMA, SIGMA2, MR, YEARLY_RM, COST)

    # Converting RF and RM to monthly rates
    RM = np.exp(YEARLY_RM/12) -1
    RF = np.exp(YEARLY_RF/12) -1

    # Creating list of arguments
    a = [[investments], [sim_type], random_seeds, [1],
         [RF], [RM], [PI_RM], [PI_RF], [PAY_TAXES], [SEED_INDEX], [COST]]


    comb_args = tuple(product(*a))

    with Pool() as p:
        res = p.starmap(main, comb_args, 2)

    dfs = pd.concat(res)
    if SEED_INDEX:
        dfs.index = dfs.index.set_levels(
            [dfs.index.levels[0], pd.date_range(start="2020-01-01", freq='MS', periods=YEARS * 12 + 1)])
    else:
        multi = pd.Index(pd.date_range(start="2020-01-01", freq='MS', periods=YEARS * 12 + 1))
        for i in range(len(random_seeds) - 1):
            multi = multi.append(pd.date_range(start="2020-01-01", freq='MS', periods=YEARS * 12 + 1))
        dfs.index = multi

    return dfs


if __name__ == "__main__":

    import cProfile, pstats
    import datetime as dt
    profiler = cProfile.Profile()
    profiler.enable()

    #fetch_returns('garch', range(100))
    data = pd.read_csv('shiller_data.txt', sep="\t", index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index.date)
    data['sp_return'] = data['sp'].pct_change()
    begin = dt.date(1871, 1, 1).strftime('%Y-%m-%d')
    end = dt.date(1921, 1, 1).strftime('%Y-%m-%d')
    returns = data.loc[begin:end, 'sp_return'].values
    rf = data.loc[begin:end, 'long_rf'].values/100
    rm = rf + 0.02
    tic = time.perf_counter()
    #shil = fetch_returns_shiller(returns, rf, rm)

    #shil.index = pd.date_range(begin, end, freq='MS')
    # plt.plot(shil['dual_phase'])
    # plt.plot(shil['single_phase'])
    # plt.plot(shil['100'])
    # plt.plot(shil['9050'])
    #plt.plot(shil['pi_rm'])
    #plt.plot(shil['pi_rf'])
    #plt.legend(['dual_phase', 'single_phase', '100', '9050'])
    #plt.legend(['pi_rm', 'pi_rf'])
    test = fetch_returns('garch', range(3), YEARLY_RF=0.02, YEARLY_RM=0.023)
    print(test.loc[(2, slice(None)), ['interest']].head(100))
    #plt.plot(test.loc[(2, slice(None)), ['interest']])
    plt.show()
    toc = time.perf_counter()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.reverse_order()
    #stats.print_stats()
    print(f"Script took {toc - tic:0.5f} seconds")
    #plt.show()
    # tic = time.perf_counter()
    # test = fetch_returns('garch', range(500), PAY_TAXES=False)
    # test2 = fetch_returns('garch', range(500), PAY_TAXES=True)
    # toc = time.perf_counter()
    # print(f"Script took {toc - tic:0.5f} seconds")
    # test = test.groupby(level=0).mean()
    # test2 = test2.groupby(level=0).mean()
    #interest = (test.interest*12/test.total_debt).fillna(value=0)
    #print(interest, test.total_debt)
    #plt.plot(test['tv_u'] - test['100'])
    #plt.plot(test2['tv_u'] - test2['100'])
    #plt.plot(test2['100'])
    #plt.plot(test2['tv_u'])

    #plt.plot(test['9050'])
    #plt.show()
