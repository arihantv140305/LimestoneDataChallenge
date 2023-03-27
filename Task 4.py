import torch.nn as nn
import torch 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import linear_model
from math import fabs

data = pd.read_csv(r'./index_predictions.csv')
df = pd.DataFrame(data)
corrM = df.corr()
print(corrM)


