import torch.nn as nn
import torch 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import linear_model

data = pd.read_csv(r'./data_challenge_stock_prices_returns_inbps.csv')
df = pd.DataFrame(data)
g1 = [0,1,4,10,11,14,18,20,21,22,25,26,38,39,43,45,49,54,67,75,78,79,82,88,90]
n1 = df.to_numpy()
n2 = np.zeros((199999, 25))

i = 0
data = pd.read_csv(r'./index_returns.csv')
df = pd.DataFrame(data)
index = df.to_numpy()[:, 1:]   #Slicing timestamp column
index = index[:, i]            #choosing index id = i

count = 0
for i in g1 : 
    n2[:, count] = n1[:, i]
    count+=1

regr = linear_model.LinearRegression()
regr.fit(n2, index)
prediction = regr.predict(n2[:, :])
#print(prediction)
print(np.corrcoef(np.vstack((prediction, index[:]))))
#plt.plot(prediction[:50], color = "orange")
#plt.plot(index[:50], color = "blue")
#plt.show()

print("Regression coefficients are: ", regr.coef_)
print("e_i is: ", regr.intercept_)
