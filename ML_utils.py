import numpy as np
import torch

def wrap_data(x_train, num_t):
    x_list = []
    lenth = len(x_train) - num_t
    for i in range(num_t):
        x_list.append(x_train[i:lenth+i+1])
    x_list = np.stack(x_list, axis=1)
    return x_list

def wrap_torch(x_train, num_t):
    x_list = []
    lenth = len(x_train) - num_t
    for i in range(num_t):
        x_list.append(x_train[i:lenth+i+1])
    x_list = torch.stack(x_list, axis=1)
    return x_list

def unwrap_data(x0, offset):
    x0_eco = x0[:-1,0]
    for idx in range(offset):
        x0_eco = np.concatenate([x0_eco, x0[-1,idx:idx+1]], axis=0)
    return x0_eco

def unwrap_torch(x0, offset):
    x0_eco = x0[:-1,0]
    for idx in range(offset):
        x0_eco = torch.cat([x0_eco, x0[-1,idx:idx+1]], dim=0)
    return x0_eco


