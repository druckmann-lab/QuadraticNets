import numpy as np
import torch
from scipy.integrate import odeint
#import matplotlib.pyplot as plt
import scipy
import scipy.spatial

def Veronese(x,d,k,T):
        X = torch.Tensor(T,d**k)
        for t in range(T):
            temp = x[t,:].squeeze()
            len = d
            # print(temp)
            for i in range(k-1):
                len = len*d
                temp = torch.ger(temp,x[t,:].squeeze()).view(len)
                # print(temp)
            X[t,:] = temp
        return X

def BestLinearMap(y,x,d,k,NumSamples):
    y = y - np.mean(y)
    z = torch.from_numpy(x.transpose())
    X = Veronese(z,d,k,NumSamples).numpy()
    m, c, r, s  = np.linalg.lstsq(X,np.transpose(y),rcond = None)
    relError = ((np.transpose(y)-np.matmul(X,m))**2).mean()/(np.transpose(y)**2).mean()
    return relError

def generate_normal(d = 10, r = 5, NumSamples = 1000, k = 2, snr = 0):
    x = np.random.randn(d,NumSamples)
    Q = np.random.randn(r,d)
    Lambda = np.random.randn(1,r)
    Qxk = np.matmul(Q,x)**k
    y = np.matmul(Lambda,Qxk)
    relError = BestLinearMap(y,x,d,k,NumSamples)
    return Q, Lambda, y, x, relError

def generate_random(d = 10, r = 5, NumSamples = 1000, k = 2):
    x = np.random.randn(d,NumSamples)
    y = np.random.randn(1,NumSamples)
    relError = BestLinearMap(y,x,d,k,NumSamples)
    #linear regression here
    return y, x, relError

def generate_identity(d = 10, r = 5, NumSamples = 1000, k = 2, snr = 0):
    x = np.random.randn(r,NumSamples)
    Q = np.identity(r)
    Qxk = np.matmul(Q,x)**k
    Lambda = np.ones((1,r))
    Lambda[0,np.random.binomial(1,.5,r)] = -1
    Lambda = Lambda + 100
    y = np.matmul(Lambda,Qxk)
    relError = BestLinearMap(y,x,d,k,NumSamples)
    return Q, Lambda, y, x, relError

def gen_data(y,NumNeurons,snr):
    y = y/np.linalg.norm(y)
    N = np.random.randn(1,NumSamples).astype(np.float32)
    N = N/np.linalg.norm(N)*np.sqrt(NumSamples/snr)
    y = y + N
    return y 
