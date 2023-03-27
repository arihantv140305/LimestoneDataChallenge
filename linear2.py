import torch.nn as nn
import torch 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import linear_model
from math import fabs

def func(a) : 
    b = np.array(a)
    for i in range(len(a)) : 
        b[i] = a[i]
    return b

data = pd.read_csv(r'./group1.csv')
df = pd.DataFrame(data)

"""g1 = [0,1,4,10,11,14,18,20,21,22,25,26,38,39,43,45,49,54,67,75,78,79,82,88,90]
g2 = [2,3,6,7,9,16,17,19,23,28,32,33,42,47,51,53,58,60,63,66,72,81,87,91,98]
g3 = [5,13,15,24,31,34,35,37,48,56,61,2,64,65,68,70,76,83,85,86,89,93,94,95,97]
g4 = [8,12,27,29,30,36,40,41,44,46,50,52,55,57,59,69,71,73,74,77,80,84,92,96,99]"""

list1 = [2,3,6,7,9,16,17,19,23,28,32,33,42,47,51,53,58,60,63,66,72,81,87,91,98]
n1 = df.to_numpy()[:199999, :  ]


index_id = 5 #g1 : 0-weighted average,2 ,7-min of return
              #g2 : 3, 6-weighted average, 8 -min of returns
              #g3 : 1-min of returns, 5, 6? , 10+- , 
              #g4 : 4-weighted average, 9+- , 14?
data = pd.read_csv(r'./index_returns.csv')
df = pd.DataFrame(data)
index = df.to_numpy()[:, 1:]   #Slicing timestamp column 
index = index[:, index_id]            #choosing index id = i



regr = linear_model.LinearRegression()
regr.fit(n1, index)
prediction = regr.predict(n1)
#print(prediction)
print(f"Corrcoef for index {index_id} is: " , np.corrcoef(np.vstack((prediction, index[:])))[0][1])
#plt.plot(prediction[:50], color = "orange")
#plt.plot(index[:50], color = "blue")
#plt.show()
list1 = list()
print("Regression coefficients are: ", regr.coef_)
for i in range(len(regr.coef_)):
    if fabs(regr.coef_[i]) > 0.035:
        list1.append(i)
print(list1)
print("e_i is: ", regr.intercept_)
