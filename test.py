import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_sector(epsilon):
    data = pd.read_csv(r'data_challenge_stock_prices.csv')
    df = pd.DataFrame(data)
    buff = []
    sec_count = 0
    for i in range(100):
        for j in range(100):
            if i>j:
                a1 = df['{}'.format(i)].to_numpy()
                b1 = df['{}'.format(j)].to_numpy()
                if np.std(np.absolute(a1-b1)) < epsilon:
                    if (i not in buff) and (j not in buff):
                      sec_count+= 1
                      buff.append(i)
                      buff.append(j)
    return sec_count
x = []
y = []
i=0
while i<20:
    x.append(i+1)
    y.append(get_sector(i+1))
    i+=0.5
plt.plot(x, y)
plt.show()
