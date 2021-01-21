import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

spx = pd.read_csv('^GSPC.csv', index_col=0)

spx.index = pd.to_datetime(spx.index)
spx = spx.asfreq(freq='BMS', method='bfill')
#spx['monthly_chg'] = spx['Adj Close'].pct_change()

spx['monthly_chg'] = spx['Adj Close'].pct_change()


trend = 1.07**(1/12)
std = spx['monthly_chg'].std()
market = np.empty([12*60, 100])

for j in range(100):
    market[0, j] = 100
    for i in range(1, 12*60):
        rnd = random.normalvariate(0, std)
        market[i, j] = (market[i-1, j]*trend+rnd*market[i-1, j])


plt.plot(market)
#plt.yscale('log')
plt.show()