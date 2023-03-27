import csv
import seaborn as sns
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt


epsilon = 5
data = pd.read_csv(r'./CorrelationMatrix_returns.csv')
df = pd.DataFrame(data)
#buff = np.zeros((101, 100))


x = []
count=0
buff = df.to_numpy()[:, 1:]
b = np.zeros((100,1))
def maxi(arr):
    ans = 0
    for i in len(arr):
        if arr[i] < 1:
            ans = max(ans, arr[i])
    return ans
def search(a) :
    print(a, end = ": ")
    count = 0 
    if a not in x :
        b[a] = 1 
        x.append(a)
        for i in range(100) :
            if (buff[a][i] > 0.12 and buff[a][i] < 1)  :
                #print(i, "called inside search of ", a)
                print(i, end = " ")
                count+=1
#print(buff)
    print("", end = "\n")
    return count

for i in range(100) :
    search(i)
#print(sorted(x))
#print(len(x))