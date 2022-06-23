import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt
import time

device ='cuda' if torch.cuda.is_available() else 'cpu'

class LayeredModel(nn.Module):
    def __init__(self, in_var = 1, features = 40, layer_count = 1, p = 0.35, act = nn.ReLU):
        super().__init__()
        self.p = p
        self.act = act
        layers = [nn.Linear(in_var, features),]
        for i in range(layer_count):
            layers.extend(self._layer(features))
        self.layers = nn.Sequential(*layers)
        self.layers.append(nn.Linear(features, in_var))
        
    def forward(self, x):
        return self.layers(x)

    def _layer(self, features):
        return [nn.Linear(features, features), self.act(), nn.Dropout(self.p)]

def generate_data(low = -1, high = 2, func = None, count = 10000, val_low = -3, val_high = 4):
    in_set = np.linspace(low, high, count)
    out_set = func(in_set)
    test_set = np.linspace(val_low, val_high, 1000)
    return in_set, out_set, test_set, func(test_set)

optim = torch.optim.RMSprop

def fit(func, features = 40, layer_count = 1, p = 0.35,
        batch_size = 128, epoch = 100, lr = 0.0001,
        low = -1, high = 2, count = 10000, time_limit = 200,
        val_low = -3, val_high = 4,
        activation = nn.ReLU):
    input_set, res, test_x, test_res = generate_data(low = low, high = high, func = func, count = count, val_low = val_low,
                                                     val_high = val_high)
    loss = nn.MSELoss()
    model = LayeredModel(1, features, layer_count, p, act = activation).to(device)
    optimizer = optim(model.parameters(), lr = lr)

    x_up = torch.FloatTensor(input_set, device = 'cpu').unsqueeze(1)
    y_up = torch.FloatTensor(res, device = 'cpu').unsqueeze(1)
    dataset = TensorDataset(x_up, y_up)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True)

    x_up.data.to(device) 
    y_up.data.to(device)
    losses = []
    st = time.time()
    for epochs in range(int(epoch)):
        for batch_idx, samples in enumerate(dataloader):
            x_use, y_use = samples
            pred = model(x_use.to(device))
            loss_calc = loss(pred, y_use.to(device))
            optimizer.zero_grad()
            loss_calc.backward()
            optimizer.step()
            losses.append(loss_calc.item())
        if epochs % 20 == 0:
            print( (epoch - epochs) * (time.time() - st) / (epochs + 1))
        if time.time() - st > time_limit:
            break
    print(time.time() - st)
    plt.plot(input_set, res)
    model.eval()
    with torch.no_grad():
        train_res = model(torch.tensor(input_set, device = device, dtype = torch.float32).unsqueeze(1)).squeeze().cpu().numpy()
        t_res_data = model(torch.tensor(test_x, device = device, dtype = torch.float32).unsqueeze(1)).squeeze().cpu().numpy()
        plt.plot(input_set, res, label = 'real_train')
        plt.plot(input_set, train_res , label = 'predicted_train'  )
        print(np.sum((res- train_res)**2))
        print(np.sum((test_res- t_res_data)**2))
        plt.plot(test_x, test_res, label = 'real_test')
        plt.plot(test_x, t_res_data , label = 'predicted_test' )
        plt.legend()
        plt.show()
    return model, losses
        
        
