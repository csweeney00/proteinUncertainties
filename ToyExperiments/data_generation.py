import numpy as np
import pandas as pd
import torch

def f(xs):
    return np.sin(np.power(xs, 2)) + xs*np.cos(2*xs)

def f_var(x):
    return np.sin(x)**2 + x/8+ 0.01
def f_noise(xs):
    return np.array([np.random.normal(0, f_var(2*x/3)) for x in xs])    # more complex noise variance

# Creates the dataset for the simple regression task
def unimodal_1d_regression(n, epi = 'LOW', ale = 'LOW', hetero = False):
    if not hetero:
        if ale == 'LOW':
            sigma = 0.2
        elif ale == 'HIGH':
            sigma = 2
        else:
            raise Exception("You have entered in the wrong aleatoric level.")
    else:
        sigma = np.nan

    if epi == 'LOW':
        x_values = np.random.uniform(low = -2, high = 2, size = n)
    elif epi == 'HIGH':         # Added to allow for training point generation with gaps, but not used
        x1 = np.random.uniform(low = -2, high = 0, size = 3*n//4)
        x2 = np.random.uniform(low = 1, high = 2, size = n - 3*n//4)
        x_values = np.concatenate([x1, x2])
    else:
        raise Exception("You have entered in the wrong epistemic level.")

    base_y = f(x_values)
    
    if hetero:
        noise_y = f_noise(x_values)
    else:
        noise_y = np.random.normal(0, sigma, n)

    y_values = base_y + noise_y

    return x_values, y_values

def g(ts):
    mean = []
    for t in ts:
        mean.append([5*t/4 * np.cos(t)**2, 5*t/4 * np.sin(t)**2, 5*np.cos(t)/3])
    return np.transpose(np.array(mean), (0,2,1))

# noise is generated from a multivariate gaussian
def g_noise(ts, sigma, lengthscale):
    #cov = lambda x,y : sigma**2 * np.exp(-(x-y)**2 / (2*lengthscale**2)) + np.sin(x/3)**2 * (x == y)        # RBF kernel
    cov = lambda x,y : sigma**2 * np.exp(-(x-y)**2 / (2*lengthscale**2)) + 0.2*np.abs(x-y)*(np.abs(x-y) > np.pi) + np.sin(x/3)**2 * (x == y)    # Altered RBF kernel

    noise = []
    for t in ts:
        c = np.array([[cov(t1, t2) for t2 in ts[0]] for t1 in ts[0]])
        xs = np.random.multivariate_normal(np.zeros(t.shape), c)
        ys = np.random.multivariate_normal(np.zeros(t.shape), c)
        zs = np.random.multivariate_normal(np.zeros(t.shape), c)
        noise.append([xs, ys, zs])
    
    return np.transpose(np.array(noise), (0,2,1))

# 3d as in the function line is embedded in R3
def unimodal_3d_regression(n, data_size, sigma, lengthscale, epi = 'LOW', ale = 'LOW', hetero = False, test=False):

    if epi == 'LOW':
        # We keep the sensor locations fixed for all training points
        if test:
            t_values = np.array([np.random.uniform(low=-1*np.pi, high=np.pi, size=data_size) for i in range(n)])
        else:
            t_values = np.array([np.linspace(-1*np.pi, np.pi, data_size) for i in range(n)])
    else:
        raise Exception("You have entered in the wrong epistemic level.")

    base_z = np.array(g(t_values))
    
    if hetero:
        noise_z = np.array(g_noise(t_values, sigma, lengthscale))
    else:
        raise Exception("You have entered in the wrong aleatoric level.")

    z_values = base_z + noise_z

    return t_values, z_values

# Converts the np arrays generated by the above functions to a torch
# dataset for training.
def np_to_torch_dataset(xs, ys, batch_size, scaler):
    if scaler != None:
        xs = scaler.transform(xs)
    xs = torch.FloatTensor(xs)
    ys = torch.FloatTensor(ys)

    print(xs.shape)

    data = torch.utils.data.TensorDataset(xs, ys)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader
