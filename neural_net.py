import torch.nn as nn
import torch 
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
batch_size = 2000
def get_sample(t1, t2, n2, index):
    s = t2 - t1
    inputs = torch.empty((s,25), dtype = torch.float)
    targets = torch.empty((s,1), dtype = torch.float)
    for i in range(t1, t2):
        targets[i-t1][0] = index[i]
        inputs[i-t1] = torch.from_numpy(n2[i]).type(torch.float)
    
    return (inputs, targets)


s = 1000
def get_values(batch_size):
    x = 199999
    ret = []
    while(x>189999):
        ret.append((x, x- batch_size))
        x -= batch_size
    return ret


data = pd.read_csv(r'./data_challenge_stock_prices_returns_inbps.csv')
df = pd.DataFrame(data)
g1 = [0,1,4,10,11,14,18,20,21,22,25,26,38,39,43,45,49,54,67,75,78,79,82,88,90]
n1 = df.to_numpy()
n2 = np.zeros((199999, 25))

data = pd.read_csv(r'./index_returns.csv')
df = pd.DataFrame(data)
index = df.to_numpy()
index = index[:, 1]



count = 0
for i in g1 : 
    n2[:, count] = n1[:, i]
    count+=1


input_size = 25
hidden_sizes = [4, 2]
output_size = 1# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
epochs = 50000
#inputs = torch.from_numpy(n2).type(torch.float)

sample_list = []
for t2, t1 in get_values(batch_size):
    inputs, targets = get_sample(t1, t2, n2, index)
    sample_list.append((inputs, targets))

# Train the neural network
'''
for epoch in tqdm(range(epochs)):
    for inputs, targets in sample_list:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save
'''

model = torch.jit.load('model_scripted.pt')
model.eval()


test_inputs, test_targets  = get_sample(1, 199999, n2, index)
test_outputs = model(test_inputs)

test_targets = torch.flatten(test_targets)
test_outputs = torch.flatten(test_outputs)

T = torch.stack((test_targets, test_outputs))
print(torch.corrcoef(T))
