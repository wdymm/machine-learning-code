import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

pGT = torch.Tensor([1./12, 2./12, 3./12, 3./12, 2./12, 1./12])
cGT = pGT.cumsum(dim=0)

y = torch.rand(1000).view(-1,1)
y = (y - cGT - 1e-5).lt(0).max(dim=1, keepdim=True)[1]
delta = torch.zeros(y.numel(),6).scatter(1,y,torch.ones_like(y).float())

#maximum likelihood given dataset y encoded in delta
def MaxLik(delta):
    alpha = 1
    theta = torch.randn(6)
    for iter in range(100):
        p_theta = torch.nn.Softmax(dim=0)(theta)
        g = torch.mean(p_theta-delta,0)
        theta = theta - alpha*g
        print("Diff: %f" % torch.norm(p_theta - pGT))
    
    return theta

theta = MaxLik(delta)

#reinforce with reward R
def Reinforce(R, theta=None):
    alpha = 1
    if theta is None:
        theta = torch.randn(6)
    for iter in range(10000):
        #current distribution
        p_theta = torch.nn.Softmax(dim=0)(theta)

        #sample from current distribution and compute reward
        ##############################
        ## Sample from p_theta, find the assignment delta and compute the reward
        ## for each sample
        ## Dimensions: cPT (6); y (1000x1 -> 1000x1); delta (1000x6); curReward (1000x1)
        ##############################
        cPT = p_theta.cumsum(dim=0)
        y = torch.rand(100000).view(-1,1)
        y = ((y - cGT - 1e-5).lt(0)*(-1)+1).sum(dim=1).view(-1,1)
        delta = torch.zeros(y.numel(),6).scatter(1,y,torch.ones_like(y).float())
        curReward = pGT

        #compute gradient and update
        g = torch.mean(curReward*(delta - p_theta),0)
        theta = theta + alpha*g
        print("Diff: %f" % torch.norm(p_theta - pGT))
        print(p_theta)

R = pGT
Reinforce(R, theta)
    
