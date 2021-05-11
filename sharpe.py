import numpy as np
import pandas as pd
from math import sqrt


def run_checks(df):
    assert isinstance(df.index, pd.MultiIndex)
    for col in ['dual_phase', 'single_phase', '100', '9050']:
        assert(col in df.columns)
    

def calculate_sharpe(df):
    run_checks(df)
    
    df['total_return_dual_phase'] = df.groupby(level=0, as_index=False).apply(lambda x: x.dual_phase/x.savings.cumsum()).reset_index(level=0, drop=True).pct_change().fillna(0)
    df['total_return_single_phase'] = df.groupby(level=0, as_index=False).apply(lambda x: x.single_phase/x.savings.cumsum()).reset_index(level=0, drop=True).pct_change().fillna(0)
    df['total_return_100'] = df.groupby(level=0, as_index=False).apply(lambda x: x['100']/x.savings.cumsum()).reset_index(level=0, drop=True).pct_change().fillna(0)
    df['total_return_9050'] = df.groupby(level=0, as_index=False).apply(lambda x: x['9050']/x.savings.cumsum()).reset_index(level=0, drop=True).pct_change().fillna(0)
    
    # Removes extreme pct change values
    df = df[df.total_return_dual_phase.abs() < 1] 

    sharpe_dual =   df.total_return_dual_phase.groupby(level=0).mean()/  df.total_return_dual_phase.groupby(level=0).std()*sqrt(12)
    sharpe_single = df.total_return_single_phase.groupby(level=0).mean()/df.total_return_single_phase.groupby(level=0).std()*sqrt(12)
    sharpe_100 =    df.total_return_100.groupby(level=0).mean()/         df.total_return_100.groupby(level=0).std()*sqrt(12)
    sharpe_9050 =   df.total_return_9050.groupby(level=0).mean()/        df.total_return_9050.groupby(level=0).std()*sqrt(12)
    
    
    sharpe_ratios = pd.concat([sharpe_dual, sharpe_single, sharpe_100, sharpe_9050], axis=1)
    sharpe_ratios.columns = ["Dual phase", "Single phase", "100% stocks", "Life cycle"]
    sharpe_ratios = sharpe_ratios[~sharpe_ratios.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)
    
    return sharpe_ratios


def CE(df, strategy, gamma=2, cutoff = 10000):
    input_list=df[strategy][abs(df[strategy]) > cutoff].to_list()
    len_list = len(input_list)
    return (1/len_list)**(1/(1-gamma))*sum([(x)**(1-gamma) for x in input_list])**(1/(1-gamma))


def CE_ports(df, gamma=2, cutoff = 10000, risk_premium = False):
    run_checks(df)
    
    CE_data = df[["dual_phase","single_phase","100","9050"]]
    max_date = max(df.index.levels[1])
    CE_data = CE_data.loc[(slice(None), max_date),:].reset_index()
    CE_data = CE_data.drop(["period"], axis=1).set_index("random_state")
    
    CE_list = []
    for strategy in ['dual_phase', 'single_phase', '100', '9050']:
        input_list=CE_data[strategy][abs(CE_data[strategy]) > cutoff].to_list()
        len_list = len(input_list)
        CE_val = (1/len_list)**(1/(1-gamma))*sum([(x)**(1-gamma) for x in input_list])**(1/(1-gamma))
        CE_list.append(pd.DataFrame([CE_val], columns=[strategy]))

    CE_list = pd.concat(CE_list, axis=1)
    CE_list.index = pd.Index(['Certainty Equivalent'])
    
    # Take mean of strategies
    if risk_premium:
        CE_list = (CE_data.mean()/CE_list-1)*100
    
    
    return CE_list



if __name__ == "__main__":
    print('helo world')