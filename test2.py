import torch.nn as nn
import torch 
import numpy as np
import random

def get_data(s):
    
    inputs = torch.empty((s,25), dtype = torch.float)
    targets = torch.empty((s,1), dtype = torch.float)
    for i in range(s):
        a = np.random.randint(1, 10, size = 25)
        for j in range(25):
            inputs[i][j] = a[j]
        targets[i][0] = inputs[i][0]*inputs[i][1] + inputs[i][2] + inputs[i][3]/inputs[i][4]
    inputs = inputs.type(torch.float)
    #targets = targets.type(torch.float)
    #targets = inputs.sum(dim=1, keepdim=True)
    inputs = (inputs - torch.mean(inputs))/torch.std(inputs)
    #targets = (targets - torch.mean(targets))/torch.std(targets)
    return (inputs, targets)


def function(a):
    x = 0
    for i in a:
        x = x+i
    return torch.Tensor([x])

input_size = 25
hidden_sizes = [16, 8]
output_size = 1# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size))

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
epochs = 10






# Train the neural network
for epoch in range(epochs):
    inputs, targets = get_data(1000)
    for _ in range(10000):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

test_inputs ,test_targets =get_data(10)
print(test_targets)

test_outputs = model(test_inputs)

for i in range(10):
    print(f" {test_targets[i].item():.4f}\t {test_outputs[i].item():.4f}")



