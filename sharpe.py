import numpy as np
import pandas as pd
from math import sqrt


def run_checks(df):
    assert isinstance(df.index, pd.MultiIndex)
    for col in ['dual_phase', 'single_phase', '100', '9050']:
        assert(col in df.columns)
    

def calculate_sharpe(df):
    run_checks(df)
    
    df['total_return_dual_phase'] = df.groupby(level=0, as_index=False).apply(lambda x: x.dual_phase/x.savings.cumsum()).reset_index(level=0, drop=True)
    df['total_return_single_phase'] = df.groupby(level=0, as_index=False).apply(lambda x: x.single_phase/x.savings.cumsum()).reset_index(level=0, drop=True)
    df['total_return_100'] = df.groupby(level=0, as_index=False).apply(lambda x: x['100']/x.savings.cumsum()).reset_index(level=0, drop=True)
    df['total_return_9050'] = df.groupby(level=0, as_index=False).apply(lambda x: x['9050']/x.savings.cumsum()).reset_index(level=0, drop=True)
    
    sharpe_dual = df.total_return_dual_phase.groupby(level=0).pct_change().groupby(level=0).mean()/df.total_return_dual_phase.groupby(level=0).pct_change().groupby(level=0).std()*sqrt(12)
    sharpe_single = df.total_return_single_phase.groupby(level=0).pct_change().groupby(level=0).mean()/df.total_return_single_phase.groupby(level=0).pct_change().groupby(level=0).std()*sqrt(12)
    sharpe_100 = df.total_return_100.groupby(level=0).pct_change().groupby(level=0).mean()/df.total_return_100.groupby(level=0).pct_change().groupby(level=0).std()*sqrt(12)
    sharpe_9050 = df.total_return_9050.groupby(level=0).pct_change().groupby(level=0).mean()/df.total_return_9050.groupby(level=0).pct_change().groupby(level=0).std()*sqrt(12)
    
    sharpe_ratios = pd.concat([sharpe_dual, sharpe_single, sharpe_100, sharpe_9050], axis=1)
    sharpe_ratios.columns = ["Dual phase", "Single phase", "100% stocks", "Life cycle"]
    sharpe_ratios = sharpe_ratios[~sharpe_ratios.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)
    
    return sharpe_ratios


    
if __name__ == "__main__":
    print('helo world')